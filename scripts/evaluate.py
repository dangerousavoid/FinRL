from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import TRAINED_MODEL_DIR
from finrl.plot import backtest_stats

from scripts.rl_env import INITIAL_AMOUNT, build_env

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}
AGENT_NAMES = list(MODEL_CLASSES.keys())


def buy_and_hold(trade: pd.DataFrame) -> pd.DataFrame:
    """Equity de comprar BTC na 1ª barra de trade e segurar."""
    prices = trade.groupby("date")["close"].first().sort_index()
    equity = INITIAL_AMOUNT * (prices / prices.iloc[0])
    return pd.DataFrame({"date": equity.index, "account_value": equity.values})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--agents", nargs="+", default=AGENT_NAMES, choices=AGENT_NAMES)
    args = p.parse_args()

    # NÃO usar set_index: o CSV (index=False) já produz o RangeIndex 0..N-1 que o env espera
    trade = pd.read_csv(args.trade)

    curves = {}
    for name in args.agents:
        model = MODEL_CLASSES[name].load(f"{TRAINED_MODEL_DIR}/agent_{name}")
        env = build_env(trade)
        df_acc, df_act = DRLAgent.DRL_prediction(model=model, environment=env)
        df_acc = df_acc.set_index("date")["account_value"]
        curves[name] = df_acc

        print(f"\n===== Métricas: {name.upper()} =====")
        print(backtest_stats(account_value=df_acc.reset_index()))
        df_act.to_csv(f"results/actions_{name}.csv")

    bh = buy_and_hold(trade).set_index("date")["account_value"]
    curves["buy_and_hold"] = bh
    print("\n===== Métricas: BUY & HOLD BTC =====")
    print(backtest_stats(account_value=bh.reset_index()))

    result = pd.DataFrame(curves)
    result.to_csv("results/equity_comparison.csv")

    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    plt.title("Agentes DRL vs. Buy & Hold (BTC)")
    plt.xlabel("Tempo")
    plt.ylabel("Valor da carteira ($)")
    plt.savefig("results/evaluate_result.png", dpi=150, bbox_inches="tight")
    print("\nGráfico salvo em results/evaluate_result.png")


if __name__ == "__main__":
    main()
