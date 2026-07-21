from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


class TargetWeightEnv(gym.Env):
    """Ambiente de AÇÃO CONTÍNUA de posição-alvo (Fase 8.9, Experimento 1).

    Isola o efeito do ESPAÇO DE AÇÃO. Ao contrário do StockTradingEnv (que
    opera em COTAS: compra/vende N unidades → decisões de canto), aqui a ação é
    a FRAÇÃO-ALVO do patrimônio em BTC:

        a_t ∈ Box(low, high, shape=(1,))   (default [0,1]; low=-1 habilita short no futuro)

    A cada passo, dado o peso-alvo w_t:
      1. calcula o peso atual w_{t-1} = valor_posição / patrimônio (drift do preço);
      2. rebalanceia a exposição até w_t, cobrando custo proporcional ao
         turnover |w_t − w_{t-1}| (default 10 bps, env var COST_BPS);
      3. avança UMA barra e mede a variação do patrimônio.

    Recompensa: selecionável via env var REWARD_KIND (Fase 8.9, passo 2):
      - "return" (default, SEM regressão): Δv líquido de custos (variação do
        patrimônio), como antes.
      - "diff_sharpe": Sharpe diferencial de Moody & Saffell (1998), recompensa
        incremental por passo baseada nas médias móveis exponenciais (taxa
        ETA, default 0.01) do retorno líquido de custos e do seu quadrado.
        Não aplica reward_scaling (já é uma quantidade adimensional pequena).
    Determinístico dada a seed. Sem VIX/turbulence (cripto).

    Estado (observação): [caixa_norm, peso_atual] + INDICATORS (já processadas
    como razões estacionárias pelo cdd_to_finrl). Compatível com o formato que
    train.py/DRLAgent consomem: expõe get_sb_env(), save_asset_memory() e
    save_action_memory() com a mesma assinatura do StockTradingEnv.

    NÃO subclasse do StockTradingEnv: a mecânica de rebalanceamento por fração é
    fundamentalmente diferente da de cotas, então reimplementamos o mínimo aqui
    (constituição: toda lógica nova em scripts/, nada tocado em finrl/).
    """

    metadata = {"render.modes": ["human"]}

    REWARD_KINDS = ("return", "diff_sharpe")

    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list: list[str],
        initial_amount: float = 1_000_000,
        reward_scaling: float = 1e-4,
        cost_bps: float | None = None,
        action_low: float = 0.0,
        action_high: float = 1.0,
        print_verbosity: int = 10,
        turnover_eps: float = 1e-6,
        reward_kind: str | None = None,
        eta: float | None = None,
    ):
        self.df = df
        self.tech_indicator_list = list(tech_indicator_list)
        self.initial_amount = float(initial_amount)
        self.reward_scaling = float(reward_scaling)
        # custo proporcional ao turnover: default 10 bps (0.001), override via COST_BPS
        self.cost_bps = float(os.environ.get("COST_BPS", "0.001")) if cost_bps is None else float(cost_bps)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.print_verbosity = int(print_verbosity)
        self.turnover_eps = float(turnover_eps)
        # Fase 8.9 passo 2: recompensa selecionável (default "return" == comportamento anterior)
        self.reward_kind = (
            os.environ.get("REWARD_KIND", "return") if reward_kind is None else reward_kind
        )
        if self.reward_kind not in self.REWARD_KINDS:
            raise ValueError(f"REWARD_KIND desconhecido: {self.reward_kind!r} (use um de {self.REWARD_KINDS})")
        self.eta = float(os.environ.get("ETA", "0.01")) if eta is None else float(eta)

        # ação CONTÍNUA de fração-alvo; parametrizável p/ [-1,1] no futuro (short)
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, shape=(1,), dtype=np.float32
        )
        # estado: caixa_norm + peso_atual + indicadores
        self.state_dim = 2 + len(self.tech_indicator_list)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.episode = 0
        self._seed()
        self._reset_bookkeeping()
        self.state = self._build_state()

    # ------------------------------------------------------------------ helpers
    def _reset_bookkeeping(self) -> None:
        self.cash = self.initial_amount
        self.units = 0.0  # unidades de BTC mantidas
        self.cost = 0.0
        self.trades = 0
        self.terminal = False
        self.asset_memory = [self.initial_amount]
        self.rewards_memory: list[float] = []
        self.actions_memory: list[float] = []  # peso-alvo w_t efetivamente aplicado
        self.date_memory = [self._get_date()]
        # Sharpe diferencial (Moody & Saffell): médias móveis exponenciais do
        # retorno líquido de custos (A) e do seu quadrado (B); reiniciadas a
        # cada episódio, como o resto do bookkeeping.
        self._dsr_a = 0.0
        self._dsr_b = 0.0

    def _differential_sharpe_reward(self, net_return: float) -> float:
        """Dt de Moody & Saffell (1998/2001), a partir do retorno líquido de
        custos `net_return` (fração, não $) desta barra.

            Dt = (B_{t-1}·ΔA_t − 0.5·A_{t-1}·ΔB_t) / (B_{t-1} − A_{t-1}²)^(3/2)

        onde A_t = A_{t-1} + η·ΔA_t (ΔA_t = R_t − A_{t-1}) e, analogamente,
        B_t = B_{t-1} + η·ΔB_t (ΔB_t = R_t² − B_{t-1}). O denominador é a
        variância corrente elevada a 3/2; nos primeiros passos (variância
        ainda não positiva) devolve 0.0 em vez de propagar um valor
        instável/NaN.
        """
        a_prev, b_prev = self._dsr_a, self._dsr_b
        delta_a = net_return - a_prev
        delta_b = net_return * net_return - b_prev
        variance = b_prev - a_prev * a_prev
        if variance <= 1e-12:
            dt = 0.0
        else:
            dt = (b_prev * delta_a - 0.5 * a_prev * delta_b) / (variance**1.5)
        self._dsr_a = a_prev + self.eta * delta_a
        self._dsr_b = b_prev + self.eta * delta_b
        return float(dt)

    def _get_date(self):
        return self.data.date

    def _price(self) -> float:
        return float(self.data.close)

    def _total_asset(self, price: float) -> float:
        return self.cash + self.units * price

    def _current_weight(self, price: float) -> float:
        v = self._total_asset(price)
        return (self.units * price) / v if v > 0 else 0.0

    def _build_state(self) -> np.ndarray:
        price = self._price()
        v = self._total_asset(price)
        cash_norm = self.cash / v if v > 0 else 1.0
        weight = self._current_weight(price)
        indicators = [float(self.data[tech]) for tech in self.tech_indicator_list]
        return np.array([cash_norm, weight] + indicators, dtype=np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self._reset_bookkeeping()
        self.episode += 1
        self.state = self._build_state()
        return self.state, {}

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            if self.episode % self.print_verbosity == 0:
                end_total_asset = self._total_asset(self._price())
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {end_total_asset - self.asset_memory[0]:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print("=================================")
            return self.state, self.reward, self.terminal, False, {}

        # peso-alvo desejado (clip ao espaço de ação)
        w_target = float(np.clip(np.asarray(actions).ravel()[0], self.action_low, self.action_high))

        price = self._price()
        v_begin = self._total_asset(price)
        w_prev = self._current_weight(price)

        # rebalanceamento: custo proporcional ao turnover |w_t - w_{t-1}|
        turnover = abs(w_target - w_prev)
        cost = self.cost_bps * turnover * v_begin
        v_after_cost = v_begin - cost
        if turnover > self.turnover_eps:
            self.cost += cost
            self.trades += 1

        # aplica o peso-alvo sobre o patrimônio líquido de custo
        target_pos_value = w_target * v_after_cost
        self.units = target_pos_value / price if price > 0 else 0.0
        self.cash = v_after_cost - target_pos_value

        self.actions_memory.append(w_target)

        # avança uma barra
        self.day += 1
        self.data = self.df.loc[self.day, :]
        price_next = self._price()
        v_end = self._total_asset(price_next)

        self.asset_memory.append(v_end)
        self.date_memory.append(self._get_date())

        # retorno líquido de custos desta barra (v_begin já reflete o turnover
        # pago via v_after_cost, usado para dimensionar a posição)
        net_return = (v_end - v_begin) / v_begin if v_begin > 0 else 0.0

        if self.reward_kind == "diff_sharpe":
            reward = self._differential_sharpe_reward(net_return)
            self.rewards_memory.append(reward)
            self.reward = reward  # já adimensional; reward_scaling não se aplica
        else:
            # recompensa: variação do patrimônio líquida de custos (sem risco)
            reward = v_end - v_begin
            self.rewards_memory.append(reward)
            self.reward = reward * self.reward_scaling

        self.state = self._build_state()
        return self.state, self.reward, self.terminal, False, {}

    def render(self, mode="human", close=False):
        return self.state

    # -------------------------------------------------- compat. com DRLAgent/8.5
    def save_asset_memory(self) -> pd.DataFrame:
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})

    def save_action_memory(self) -> pd.DataFrame:
        # o peso É a própria ação (w_t); signals.py lê direto, sem reconstruir de cotas
        return pd.DataFrame({"date": self.date_memory[:-1], "actions": self.actions_memory})

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
