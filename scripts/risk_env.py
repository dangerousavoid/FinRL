from __future__ import annotations

import os

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class RiskAwareStockTradingEnv(StockTradingEnv):
    """StockTradingEnv (Fase 8.8) com recompensa ajustada a risco.

    Sobrescreve só o cálculo da recompensa: r = Δv − λ·ΔDD, onde Δv é o
    delta de patrimônio já calculado pelo StockTradingEnv original e ΔDD é o
    aumento do drawdown nesta barra: max(0, (pico_histórico − valor) −
    drawdown_anterior).

    λ vem da env var RISK_LAMBDA (float, default 0.0). Com λ=0 esta classe
    não toca em reward/rewards_memory — reproduz exatamente o env base
    (ver scripts/test_risk_env.py).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_lambda = float(os.environ.get("RISK_LAMBDA", "0.0"))
        self._peak_asset = self.asset_memory[0]
        self._prev_drawdown = 0.0

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._peak_asset = self.asset_memory[0]
        self._prev_drawdown = 0.0
        return result

    def step(self, actions):
        state, reward, terminal, truncated, info = super().step(actions)

        # terminal=True aqui significa que esta chamada foi o retorno antecipado
        # de fim de episódio do StockTradingEnv (nenhum Δv novo nesta chamada,
        # rewards_memory não foi atualizado) — nada a ajustar.
        if not terminal and self.risk_lambda != 0.0:
            end_total_asset = self.asset_memory[-1]
            self._peak_asset = max(self._peak_asset, end_total_asset)
            drawdown = max(0.0, self._peak_asset - end_total_asset)
            delta_dd = max(0.0, drawdown - self._prev_drawdown)
            self._prev_drawdown = drawdown

            raw_delta_v = self.rewards_memory[-1]  # Δv bruto (pré reward_scaling)
            adjusted_raw = raw_delta_v - self.risk_lambda * delta_dd
            self.rewards_memory[-1] = adjusted_raw
            reward = adjusted_raw * self.reward_scaling
            self.reward = reward

        return state, reward, terminal, truncated, info
