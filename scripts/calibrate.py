from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from stable_baselines3 import PPO

from finrl.config import PPO_PARAMS

from scripts.train import build_vec_env

N_ENVS = int(os.environ.get("N_ENVS", "1"))


def main() -> None:
    p = argparse.ArgumentParser(description="Treina um PPO descartável para medir passos/segundo agregado.")
    p.add_argument("--train", default="train_data.csv")
    p.add_argument("--calib-steps", type=int, default=20_000)
    args = p.parse_args()

    # NÃO usar set_index: o CSV (index=False) já produz o RangeIndex 0..N-1 que o env espera
    train = pd.read_csv(args.train)
    print(f"N_ENVS={N_ENVS}")
    env_train = build_vec_env(train, N_ENVS)

    model = PPO(policy="MlpPolicy", env=env_train, **PPO_PARAMS)

    # total_timesteps do SB3 já é a soma entre os N_ENVS envs (não por env), então
    # calib_steps / elapsed é diretamente o throughput AGREGADO.
    start = time.time()
    model.learn(total_timesteps=args.calib_steps)
    elapsed = time.time() - start

    steps_per_second = args.calib_steps / elapsed if elapsed > 0 else float("inf")
    print(f"calibração: {args.calib_steps:,} passos (agregados, N_ENVS={N_ENVS}) em {elapsed:.1f}s")
    print(f"STEPS_PER_SECOND={steps_per_second:.4f}")


if __name__ == "__main__":
    main()
