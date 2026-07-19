from __future__ import annotations

import argparse

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, RESULTS_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

AGENT_NAMES = ["a2c", "ddpg", "ppo", "td3", "sac"]


def build_env(train: pd.DataFrame) -> StockTradingEnv:
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }
    # turbulence_threshold=None (default): cripto não tem vix/turbulence
    return StockTradingEnv(df=train, **env_kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train_data.csv")
    p.add_argument("--agents", nargs="+", default=AGENT_NAMES, choices=AGENT_NAMES)
    p.add_argument("--total-timesteps", type=int, default=20_000)
    args = p.parse_args()

    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

    train = pd.read_csv(args.train)  # NÃO usar set_index: o CSV (index=False) já
                                      # produz o RangeIndex 0..N-1 que o env espera
    e_train_gym = build_env(train)
    env_train, _ = e_train_gym.get_sb_env()

    for name in args.agents:
        agent = DRLAgent(env=env_train)
        model = agent.get_model(name)
        logger = configure(f"{RESULTS_DIR}/{name}", ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        print(f"\n===== Treinando {name.upper()} ({args.total_timesteps} timesteps) =====")
        trained = agent.train_model(model=model, tb_log_name=name, total_timesteps=args.total_timesteps)
        trained.save(f"{TRAINED_MODEL_DIR}/agent_{name}")
        print(f"Modelo salvo em {TRAINED_MODEL_DIR}/agent_{name}")


if __name__ == "__main__":
    main()
