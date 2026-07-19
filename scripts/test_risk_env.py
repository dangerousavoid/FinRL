"""Teste rápido (Fase 8.8): RISK_LAMBDA=0 deve reproduzir exatamente o
StockTradingEnv original. Rodar com `pytest scripts/test_risk_env.py` ou
diretamente com `python scripts/test_risk_env.py`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from scripts.risk_env import RiskAwareStockTradingEnv

N_STEPS = 30


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = N_STEPS + 5
    dates = pd.date_range("2024-01-01", periods=n, freq="1h").strftime("%Y-%m-%d %H:%M:%S")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(1, 10, n),
            "tic": "BTCUSDT",
            "macd": rng.normal(0, 1, n),
            "rsi_14": rng.uniform(20, 80, n),
        }
    ).reset_index(drop=True)


def _env_kwargs(df: pd.DataFrame) -> dict:
    tech = ["macd", "rsi_14"]
    stock_dim = 1
    return dict(
        turbulence_threshold=None,
        hmax=15,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        state_space=1 + 2 * stock_dim + len(tech) * stock_dim,
        stock_dim=stock_dim,
        tech_indicator_list=tech,
        action_space=stock_dim,
        reward_scaling=1e-4,
    )


def _run(env, actions_seq):
    env.reset()
    rewards = []
    for actions in actions_seq:
        _, reward, terminal, _, _ = env.step(np.array(actions, dtype=np.float32))
        rewards.append(reward)
        if terminal:
            break
    return rewards, env.asset_memory


def test_risk_lambda_zero_matches_base_env():
    os.environ["RISK_LAMBDA"] = "0.0"
    df = _make_df()
    kwargs = _env_kwargs(df)

    rng = np.random.default_rng(42)
    actions_seq = [rng.uniform(-1, 1, 1) for _ in range(N_STEPS)]

    base_rewards, base_assets = _run(StockTradingEnv(df=df, **kwargs), actions_seq)
    risk_rewards, risk_assets = _run(RiskAwareStockTradingEnv(df=df, **kwargs), actions_seq)

    assert base_rewards == risk_rewards
    assert base_assets == risk_assets


def test_risk_lambda_positive_penalizes_drawdown():
    df = _make_df()
    kwargs = _env_kwargs(df)
    rng = np.random.default_rng(42)
    actions_seq = [rng.uniform(-1, 1, 1) for _ in range(N_STEPS)]

    os.environ["RISK_LAMBDA"] = "0.0"
    base_rewards, _ = _run(RiskAwareStockTradingEnv(df=df, **kwargs), actions_seq)

    os.environ["RISK_LAMBDA"] = "0.5"
    penalized_rewards, _ = _run(RiskAwareStockTradingEnv(df=df, **kwargs), actions_seq)

    assert sum(penalized_rewards) <= sum(base_rewards)
    assert penalized_rewards != base_rewards


if __name__ == "__main__":
    test_risk_lambda_zero_matches_base_env()
    test_risk_lambda_positive_penalizes_drawdown()
    print("OK: RiskAwareStockTradingEnv com RISK_LAMBDA=0 reproduz o StockTradingEnv original.")
