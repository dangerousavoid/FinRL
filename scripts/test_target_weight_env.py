"""Teste rápido (Fase 8.9, passo 2): REWARD_KIND=return deve reproduzir
exatamente o comportamento anterior (SEM regressão) e REWARD_KIND=diff_sharpe
deve calcular o Sharpe diferencial sem erro/NaN. Rodar com
`pytest scripts/test_target_weight_env.py` ou diretamente com
`python scripts/test_target_weight_env.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.target_weight_env import TargetWeightEnv

N_STEPS = 40


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


def _run(env, actions_seq):
    env.reset()
    rewards = []
    for actions in actions_seq:
        _, reward, terminal, _, _ = env.step(np.array(actions, dtype=np.float32))
        rewards.append(reward)
        if terminal:
            break
    return rewards, env.asset_memory


def test_reward_kind_return_matches_previous_behavior():
    df = _make_df()
    tech = ["macd", "rsi_14"]
    rng = np.random.default_rng(42)
    actions_seq = [rng.uniform(0, 1, 1) for _ in range(N_STEPS)]

    env_default = TargetWeightEnv(df=df, tech_indicator_list=tech)
    env_explicit = TargetWeightEnv(df=df, tech_indicator_list=tech, reward_kind="return")

    rewards_default, assets_default = _run(env_default, actions_seq)
    rewards_explicit, assets_explicit = _run(env_explicit, actions_seq)

    assert rewards_default == rewards_explicit
    assert assets_default == assets_explicit


def test_diff_sharpe_reward_no_nan_and_deterministic():
    df = _make_df()
    tech = ["macd", "rsi_14"]
    rng = np.random.default_rng(7)
    actions_seq = [rng.uniform(0, 1, 1) for _ in range(N_STEPS)]

    env_a = TargetWeightEnv(df=df, tech_indicator_list=tech, reward_kind="diff_sharpe", eta=0.05)
    env_b = TargetWeightEnv(df=df, tech_indicator_list=tech, reward_kind="diff_sharpe", eta=0.05)

    rewards_a, _ = _run(env_a, actions_seq)
    rewards_b, _ = _run(env_b, actions_seq)

    assert rewards_a == rewards_b  # determinístico dada a mesma sequência de ações
    assert all(math.isfinite(r) for r in rewards_a)
    # com retornos não-triviais, a recompensa diferencial deve variar (não travar em 0)
    assert any(r != 0.0 for r in rewards_a)


def test_invalid_reward_kind_raises():
    df = _make_df()
    tech = ["macd", "rsi_14"]
    try:
        TargetWeightEnv(df=df, tech_indicator_list=tech, reward_kind="nao_existe")
    except ValueError:
        return
    raise AssertionError("esperava ValueError para REWARD_KIND inválido")


if __name__ == "__main__":
    test_reward_kind_return_matches_previous_behavior()
    test_diff_sharpe_reward_no_nan_and_deterministic()
    test_invalid_reward_kind_raises()
    print("OK: TargetWeightEnv REWARD_KIND=return sem regressão; diff_sharpe sem NaN e determinístico.")
