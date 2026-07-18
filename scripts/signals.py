from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import TRAINED_MODEL_DIR

from scripts.rl_env import build_env

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}


def extract_weights(trade: pd.DataFrame, agent: str, held0: float = 0.0) -> pd.DataFrame:
    """Reconstrói o contrato date->weight (Fase 8.5) a partir de um agente FinRL.

    Único ponto que conhece o formato de ações do FinRL: traduz as ações
    (cotas escaladas por hmax) em posição acumulada e depois em peso (fração
    do patrimônio), agnóstico de FinRL, para o backtest_pro.py.
    """
    model = MODEL_CLASSES[agent].load(f"{TRAINED_MODEL_DIR}/agent_{agent}")
    env = build_env(trade)
    df_acc, df_act = DRLAgent.DRL_prediction(model=model, environment=env)

    prices = trade.groupby("date")["close"].first().sort_index()

    acts = df_act.copy()
    if "date" in acts.columns:
        acts = acts.set_index("date")
    act_col = "actions" if "actions" in acts.columns else acts.columns[0]
    # env de 1 ativo: cada linha de "actions" vem como array de shape (1,) — extrai o escalar
    scalar_actions = acts[act_col].apply(lambda a: float(np.asarray(a).ravel()[0]))
    holdings = held0 + scalar_actions.cumsum()

    equity = df_acc.set_index("date")["account_value"]

    df = pd.DataFrame({"close": prices, "holdings": holdings, "equity": equity}).dropna()
    df["weight"] = (df["holdings"] * df["close"] / df["equity"]).clip(-1, 1)
    return df.reset_index().rename(columns={"index": "date"})[["date", "weight"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--agent", default="ppo", choices=list(MODEL_CLASSES))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    trade = pd.read_csv(args.trade)
    w = extract_weights(trade, args.agent)
    out = args.out or f"results/weights_{args.agent}.csv"
    w.to_csv(out, index=False)
    print(f"contrato salvo em {out}  ({len(w)} barras, peso médio={w.weight.mean():.3f})")


if __name__ == "__main__":
    main()
