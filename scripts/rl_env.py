from __future__ import annotations

import pandas as pd

from finrl.config import INDICATORS

from scripts.risk_env import RiskAwareStockTradingEnv
from scripts.target_weight_env import TargetWeightEnv

HMAX = 15
INITIAL_AMOUNT = 1_000_000
TIC = "BTCUSDT"

STOCKTRADING = "stocktrading"
TARGET_WEIGHT = "target_weight"
ENV_KINDS = (STOCKTRADING, TARGET_WEIGHT)


def env_kwargs(df: pd.DataFrame) -> dict:
    stock_dim = len(df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    # turbulence_threshold=None: cripto não tem vix/turbulence
    return dict(
        turbulence_threshold=None,
        hmax=HMAX,
        initial_amount=INITIAL_AMOUNT,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        state_space=state_space,
        stock_dim=stock_dim,
        tech_indicator_list=INDICATORS,
        action_space=stock_dim,
        reward_scaling=1e-4,
    )


def build_env(df: pd.DataFrame, env_kind: str = STOCKTRADING):
    """Fábrica de ambiente, roteada por `env_kind` (Fase 8.9).

    - "stocktrading" (default, SEM regressão): RiskAwareStockTradingEnv, subclasse
      aditiva do StockTradingEnv que só sobrescreve a recompensa; com RISK_LAMBDA=0
      (default) reproduz exatamente o env original — seguro usar incondicionalmente.
    - "target_weight" (Fase 8.9, Exp. 1): TargetWeightEnv, ação CONTÍNUA de
      fração-alvo do patrimônio. Muda SÓ o espaço de ação; recompensa segue
      sendo Δv líquido de custos.
    """
    if env_kind == TARGET_WEIGHT:
        return TargetWeightEnv(
            df=df,
            tech_indicator_list=INDICATORS,
            initial_amount=INITIAL_AMOUNT,
        )
    if env_kind == STOCKTRADING:
        return RiskAwareStockTradingEnv(df=df, **env_kwargs(df))
    raise ValueError(f"env_kind desconhecido: {env_kind!r} (use um de {ENV_KINDS})")
