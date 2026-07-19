from __future__ import annotations

import pandas as pd

from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

HMAX = 15
INITIAL_AMOUNT = 1_000_000
TIC = "BTCUSDT"


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


def build_env(df: pd.DataFrame) -> StockTradingEnv:
    return StockTradingEnv(df=df, **env_kwargs(df))
