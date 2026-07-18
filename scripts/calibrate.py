from __future__ import annotations

import argparse
import time

import pandas as pd
from stable_baselines3 import PPO

from finrl.config import PPO_PARAMS

from scripts.rl_env import build_env


def main() -> None:
    p = argparse.ArgumentParser(description="Treina um PPO descartável para medir passos/segundo.")
    p.add_argument("--train", default="train_data.csv")
    p.add_argument("--calib-steps", type=int, default=20_000)
    args = p.parse_args()

    # NÃO usar set_index: o CSV (index=False) já produz o RangeIndex 0..N-1 que o env espera
    train = pd.read_csv(args.train)
    e_train_gym = build_env(train)
    env_train, _ = e_train_gym.get_sb_env()

    model = PPO(policy="MlpPolicy", env=env_train, **PPO_PARAMS)

    start = time.time()
    model.learn(total_timesteps=args.calib_steps)
    elapsed = time.time() - start

    steps_per_second = args.calib_steps / elapsed if elapsed > 0 else float("inf")
    print(f"calibração: {args.calib_steps:,} passos em {elapsed:.1f}s")
    print(f"STEPS_PER_SECOND={steps_per_second:.4f}")


if __name__ == "__main__":
    main()
