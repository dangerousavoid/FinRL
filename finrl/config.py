# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d %H:%M:%S' (timestamp completo — dados BTC intradiários)
# Fase 8.8: timeframe 1h. Treino contém o bear de 2022; validação = 2024 inteiro;
# trade (held-out) começa em 2025 e cobre o crash de abr/2025 fora da amostra.
TRAIN_START_DATE = "2020-01-01 00:00:00"
TRAIN_END_DATE = "2023-12-31 23:59:00"

VALIDATION_START_DATE = "2024-01-01 00:00:00"
VALIDATION_END_DATE = "2024-12-31 23:59:00"

TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-03-20"

TRADE_START_DATE = "2025-01-01 00:00:00"
# None = detectado dinamicamente por scripts/cdd_to_finrl.py como a última data
# disponível após concatenar os CSVs de data/raw/ — não deixar hard-coded aqui,
# pois o histórico da CryptoDataDownload cresce a cada novo arquivo baixado.
TRADE_END_DATE = None

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
# Fase 8.8: timeframe 1h — janelas em barras de 1h (24 barras = 1 dia, 168 = 1
# semana). Cobre tendência (macd), momentum (rsi_14, wr_14), volatilidade/regime
# (atr_14, close_24_mstd, adx) e volume (vr_26, mfi_14) — os dois buracos do
# set anterior (5min). close_24_sma/close_168_sma/atr_14/close_24_mstd são
# convertidos para formas estacionárias em scripts/cdd_to_finrl.py.
INDICATORS = [
    "macd",
    "rsi_14",
    "wr_14",
    "atr_14",
    "close_24_mstd",
    "adx",
    "vr_26",
    "mfi_14",
    "close_24_sma",
    "close_168_sma",
]


# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
