from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import TRAINED_MODEL_DIR

from scripts.rl_env import ENV_KINDS, STOCKTRADING, TARGET_WEIGHT, build_env

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}


def extract_weights(
    trade: pd.DataFrame,
    agent: str,
    held0: float = 0.0,
    model_path: str | None = None,
    env_kind: str = STOCKTRADING,
) -> pd.DataFrame:
    """Reconstrói o contrato date->weight (Fase 8.5) a partir de um agente FinRL.

    Único ponto que conhece o formato de ações do agente. Dois casos:

    - env_kind="stocktrading": as ações são COTAS escaladas por hmax; traduz em
      posição acumulada e depois em peso (fração do patrimônio).
    - env_kind="target_weight" (Fase 8.9): o peso É a PRÓPRIA ação w_t do agente
      (fração-alvo do patrimônio) — lido direto, sem reconstruir de cotas.

    model_path: caminho do modelo salvo (sem extensão .zip); default
    'trained_models/agent_<agent>'. Use para modelos de braços do ensemble
    (Fase 8.8), salvos como 'agent_<algo>_l<lambda>_s<seed>'.
    """
    path = model_path or f"{TRAINED_MODEL_DIR}/agent_{agent}"
    model = MODEL_CLASSES[agent].load(path)
    env = build_env(trade, env_kind=env_kind)
    df_acc, df_act = DRLAgent.DRL_prediction(model=model, environment=env)

    acts = df_act.copy()
    if "date" in acts.columns:
        acts = acts.set_index("date")
    act_col = "actions" if "actions" in acts.columns else acts.columns[0]
    # env de 1 ativo: cada linha de "actions" vem como escalar ou array shape (1,)
    scalar_actions = acts[act_col].apply(lambda a: float(np.asarray(a).ravel()[0]))

    if env_kind == TARGET_WEIGHT:
        # o peso é a própria ação do agente; nada a reconstruir
        df = pd.DataFrame({"weight": scalar_actions.clip(-1, 1)})
        return df.reset_index().rename(columns={"index": "date"})[["date", "weight"]]

    prices = trade.groupby("date")["close"].first().sort_index()
    holdings = held0 + scalar_actions.cumsum()
    equity = df_acc.set_index("date")["account_value"]

    df = pd.DataFrame({"close": prices, "holdings": holdings, "equity": equity}).dropna()
    df["weight"] = (df["holdings"] * df["close"] / df["equity"]).clip(-1, 1)
    return df.reset_index().rename(columns={"index": "date"})[["date", "weight"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--env", default=STOCKTRADING, choices=list(ENV_KINDS),
                   help="espaço de ação do agente treinado: 'stocktrading' (cotas) "
                        "ou 'target_weight' (peso = própria ação)")
    p.add_argument("--agent", default="ppo", choices=list(MODEL_CLASSES))
    p.add_argument("--model-path", default=None,
                   help="caminho do modelo salvo (sem .zip); default: trained_models/agent_<agent>")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    trade = pd.read_csv(args.trade)
    w = extract_weights(trade, args.agent, model_path=args.model_path, env_kind=args.env)
    out = args.out or f"results/weights_{args.agent}.csv"
    w.to_csv(out, index=False)
    print(f"contrato salvo em {out}  ({len(w)} barras, peso médio={w.weight.mean():.3f})")


if __name__ == "__main__":
    main()
