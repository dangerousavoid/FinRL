from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd
import torch
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from finrl.config import A2C_PARAMS, PPO_PARAMS, RESULTS_DIR, SAC_PARAMS, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories

from scripts.rl_env import build_env

MODEL_CLASSES = {"ppo": PPO, "a2c": A2C, "sac": SAC}
MODEL_PARAMS = {"ppo": PPO_PARAMS, "a2c": A2C_PARAMS, "sac": SAC_PARAMS}

CHECKPOINT_FREQ = 25_000
N_ENVS = int(os.environ.get("N_ENVS", "1"))
# Fase 8.8: total_timesteps default = PASSADAS x nº de linhas do train (~15 passadas
# pelos dados); override explícito via env var PASSADAS ou --total-timesteps.
PASSADAS = float(os.environ.get("PASSADAS", "15"))


def make_env(train_df: pd.DataFrame, rank: int, seed: int | None = None):
    """Fábrica: cada processo do SubprocVecEnv recebe uma instância independente do env."""

    def _init():
        from scripts.rl_env import build_env as _build_env

        env = _build_env(train_df)
        env.reset(seed=(seed or 0) + rank)
        return env

    return _init


def build_vec_env(train_df: pd.DataFrame, n_envs: int, seed: int | None = None):
    if n_envs <= 1:
        # caminho single-env (Codespace / debug) — sem regressão em relação ao anterior
        e_train_gym = build_env(train_df)
        if seed is not None:
            e_train_gym.reset(seed=seed)
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
    venv = SubprocVecEnv([make_env(train_df, i, seed=seed) for i in range(n_envs)], start_method="fork")
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
    p.add_argument("--algo", default="ppo", choices=list(MODEL_CLASSES))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="default: PASSADAS (env var, default 15) x nº de linhas do --train")
    p.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    p.add_argument("--checkpoint-prefix", default=None,
                   help="default: '<algo>_checkpoint' (dê um valor único por braço do ensemble)")
    p.add_argument("--model-name", default=None, help="default: 'agent_<algo>'")
    args = p.parse_args()

    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

    # NÃO usar set_index: o CSV (index=False) já produz o RangeIndex 0..N-1 que o env espera
    train = pd.read_csv(args.train)
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else int(PASSADAS * len(train))
    checkpoint_prefix = args.checkpoint_prefix or f"{args.algo}_checkpoint"
    model_name = args.model_name or f"agent_{args.algo}"

    print(f"N_ENVS={N_ENVS}  algo={args.algo}  seed={args.seed}  total_timesteps={total_timesteps:,}")
    env_train = build_vec_env(train, N_ENVS, seed=args.seed)

    model_cls = MODEL_CLASSES[args.algo]
    model_params = MODEL_PARAMS[args.algo]

    ckpt_path, completed_steps = _latest_checkpoint(TRAINED_MODEL_DIR, checkpoint_prefix)
    remaining = max(total_timesteps - completed_steps, 0)

    if ckpt_path:
        print(f"RESUME: checkpoint encontrado em {ckpt_path} ({completed_steps:,} passos já treinados)")
        model = model_cls.load(ckpt_path, env=env_train)
    else:
        print(f"Nenhum checkpoint encontrado — treinando {args.algo.upper()} do zero")
        model = model_cls(
            policy="MlpPolicy",
            env=env_train,
            tensorboard_log=f"{RESULTS_DIR}/{args.algo}",
            seed=args.seed,
            **model_params,
        )

    if remaining == 0:
        print(f"total_timesteps ({total_timesteps:,}) já alcançado pelo checkpoint — pulando treino")
    else:
        # CheckpointCallback conta n_calls (1 por env.step() vetorizado), não passos totais:
        # com N_ENVS>1 cada call já vale N_ENVS passos, então dividimos para manter a
        # cadência pretendida em passos totais (aviso oficial do SB3).
        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // N_ENVS, 1),
            save_path=TRAINED_MODEL_DIR,
            name_prefix=checkpoint_prefix,
        )
        print(f"Treinando {args.algo.upper()} por mais {remaining:,} passos (alvo total: {total_timesteps:,})")
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=(ckpt_path is None))

    model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
    print(f"Modelo final salvo em {TRAINED_MODEL_DIR}/{model_name}")


if __name__ == "__main__":
    main()
