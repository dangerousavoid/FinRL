from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from finrl.config import PPO_PARAMS, RESULTS_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories

from scripts.rl_env import build_env

CHECKPOINT_PREFIX = "ppo_checkpoint"
CHECKPOINT_FREQ = 25_000


def _latest_checkpoint(model_dir: str, prefix: str) -> tuple[str | None, int]:
    """Devolve (caminho, passos) do checkpoint mais avançado, ou (None, 0) se não houver."""
    pattern = os.path.join(model_dir, f"{prefix}_*_steps.zip")
    paths = glob.glob(pattern)
    if not paths:
        return None, 0

    def steps_of(path: str) -> int:
        m = re.search(rf"{re.escape(prefix)}_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else -1

    best = max(paths, key=steps_of)
    return best, steps_of(best)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train_data.csv")
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    p.add_argument("--checkpoint-prefix", default=CHECKPOINT_PREFIX)
    p.add_argument("--model-name", default="agent_ppo")
    args = p.parse_args()

    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

    # NÃO usar set_index: o CSV (index=False) já produz o RangeIndex 0..N-1 que o env espera
    train = pd.read_csv(args.train)
    e_train_gym = build_env(train)
    env_train, _ = e_train_gym.get_sb_env()

    ckpt_path, completed_steps = _latest_checkpoint(TRAINED_MODEL_DIR, args.checkpoint_prefix)
    remaining = max(args.total_timesteps - completed_steps, 0)

    if ckpt_path:
        print(f"RESUME: checkpoint encontrado em {ckpt_path} ({completed_steps:,} passos já treinados)")
        model = PPO.load(ckpt_path, env=env_train)
    else:
        print("Nenhum checkpoint encontrado — treinando PPO do zero")
        model = PPO(policy="MlpPolicy", env=env_train, tensorboard_log=f"{RESULTS_DIR}/ppo", **PPO_PARAMS)

    if remaining == 0:
        print(f"total_timesteps ({args.total_timesteps:,}) já alcançado pelo checkpoint — pulando treino")
    else:
        callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=TRAINED_MODEL_DIR,
            name_prefix=args.checkpoint_prefix,
        )
        print(f"Treinando PPO por mais {remaining:,} passos (alvo total: {args.total_timesteps:,})")
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=(ckpt_path is None))

    model.save(f"{TRAINED_MODEL_DIR}/{args.model_name}")
    print(f"Modelo final salvo em {TRAINED_MODEL_DIR}/{args.model_name}")


if __name__ == "__main__":
    main()
