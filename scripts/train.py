from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from finrl.config import PPO_PARAMS, RESULTS_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories

from scripts.rl_env import build_env, env_kwargs

CHECKPOINT_PREFIX = "ppo_checkpoint"
CHECKPOINT_FREQ = 25_000
N_ENVS = int(os.environ.get("N_ENVS", "1"))


def make_env(train_df: pd.DataFrame, rank: int):
    """Fábrica: cada processo do SubprocVecEnv recebe uma instância independente do env."""

    def _init():
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

        env = StockTradingEnv(df=train_df, **env_kwargs(train_df))
        env.reset(seed=rank)
        return env

    return _init


def build_vec_env(train_df: pd.DataFrame, n_envs: int):
    if n_envs <= 1:
        # caminho single-env (Codespace / debug) — sem regressão em relação ao anterior
        e_train_gym = build_env(train_df)
        env_train, _ = e_train_gym.get_sb_env()
        return env_train
    # evita o processo principal competir por núcleos com os N workers do SubprocVecEnv
    # (só o processo principal roda a rede neural via torch; os workers só rodam o
    # step() em pandas) — sem isso o ganho do paralelismo é corroído pela contenção de CPU.
    torch.set_num_threads(1)
    # start_method="fork": evita reimportar stable_baselines3 (e suas dependências
    # pesadas, ex. cv2) em cada processo filho — default (forkserver/spawn) falha
    # em containers Linux headless sem libGL. "fork" também é o mais rápido no Linux
    # (destino real: Codespace/VPS), então não há trade-off aqui.
    venv = SubprocVecEnv([make_env(train_df, i) for i in range(n_envs)], start_method="fork")
    return VecMonitor(venv)  # preserva as métricas de episódio (ep_rew_mean etc.)


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
    print(f"N_ENVS={N_ENVS}")
    env_train = build_vec_env(train, N_ENVS)

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
        # CheckpointCallback conta n_calls (1 por env.step() vetorizado), não passos totais:
        # com N_ENVS>1 cada call já vale N_ENVS passos, então dividimos para manter a
        # cadência pretendida em passos totais (aviso oficial do SB3).
        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // N_ENVS, 1),
            save_path=TRAINED_MODEL_DIR,
            name_prefix=args.checkpoint_prefix,
        )
        print(f"Treinando PPO por mais {remaining:,} passos (alvo total: {args.total_timesteps:,})")
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=(ckpt_path is None))

    model.save(f"{TRAINED_MODEL_DIR}/{args.model_name}")
    print(f"Modelo final salvo em {TRAINED_MODEL_DIR}/{args.model_name}")


if __name__ == "__main__":
    main()
