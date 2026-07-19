#!/usr/bin/env bash
# Fase 8.6 — experimento sério: gera dados, calibra velocidade de treino,
# dimensiona total_timesteps por orçamento de tempo, treina PPO, avalia em
# validação e roda o backtest "pro" (Fase 8.5) contra buy-and-hold do BTC.
#
# FRESH=${FRESH:-0}: controla a etapa (a) "Começo limpo".
#   FRESH=0 (default) — preserva trained_models/ e results/, permitindo que
#                        o treino (etapa e) faça resume a partir do checkpoint.
#   FRESH=1            — esvazia trained_models/ e results/ antes de começar
#                        (preservando run.log; NÃO mexe em data/raw/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ORÇAMENTO DE TEMPO PARA O TREINO PPO, em segundos. Editável (ou via env).
ORCAMENTO_SEG="${ORCAMENTO_SEG:-10800}"  # 3 horas

CSV_DIR="${CSV_DIR:-data/raw}"
RESAMPLE="${RESAMPLE:-5min}"
N_ENVS="${N_ENVS:-1}"  # nº de envs paralelos (SubprocVecEnv); 1 = caminho single-env (Codespace)
RESULTS_DIR="results"
FRESH="${FRESH:-0}"

export N_ENVS

log() { echo "[run_experiment] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }

if [ "$FRESH" = "1" ]; then
    log "== (a) Começo limpo (FRESH=1): esvaziando conteúdo de trained_models/ e results/ (preservando run.log em uso; NÃO mexe em data/raw/) =="
    # find -mindepth 1 -delete (não rm -rf): trained_models/ e results/ são volumes Docker
    # montados na VPS — remover o próprio diretório dá "Device or resource busy".
    find trained_models -mindepth 1 -delete 2>/dev/null || true
    mkdir -p trained_models
    # preserva run.log: é o próprio log deste script (redirecionado pelo nohup) e
    # apagá-lo no meio da escrita perderia o log quando o processo terminasse.
    find "$RESULTS_DIR" -mindepth 1 ! -name 'run.log' -delete 2>/dev/null || true
    mkdir -p "$RESULTS_DIR"
else
    log "== (a) FRESH=0: preservando trained_models/ e results/ para permitir resume =="
    mkdir -p trained_models
    mkdir -p "$RESULTS_DIR"
fi

START_TS=$(python -c 'import time; print(time.time())')

log "== (b) Gerando datasets a partir de ${CSV_DIR} (resample=${RESAMPLE}) =="
python scripts/cdd_to_finrl.py --csv "$CSV_DIR" --tic BTCUSDT --resample "$RESAMPLE" \
    --train-out train_data.csv --val-out val_data.csv --trade-out trade_data.csv

N_LINHAS_TRAIN=$(($(wc -l < train_data.csv) - 1))
log "train_data.csv: ${N_LINHAS_TRAIN} linhas"

log "== (c) Calibração: 20.000 passos PPO descartáveis, cronometrando (N_ENVS=${N_ENVS}) =="
CALIB_OUT=$(python scripts/calibrate.py --train train_data.csv --calib-steps 20000)
echo "$CALIB_OUT"
STEPS_PER_SECOND=$(echo "$CALIB_OUT" | grep -oP 'STEPS_PER_SECOND=\K[0-9.]+')
log "calibração: ${STEPS_PER_SECOND} passos/segundo agregados (N_ENVS=${N_ENVS})"

log "== (d) Calculando total_timesteps (orçamento de tempo vs. 3 passadas pelos dados) =="
BUDGET_CALC=$(python - "$N_LINHAS_TRAIN" "$STEPS_PER_SECOND" "$ORCAMENTO_SEG" <<'PY'
import sys
n_linhas, steps_per_second, orcamento = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
cap_3_epocas = int(3 * n_linhas)
cap_orcamento = int(steps_per_second * orcamento)
total = min(cap_3_epocas, cap_orcamento)
print(total, cap_3_epocas, cap_orcamento)
PY
)
read -r TOTAL_TIMESTEPS CAP_EPOCAS CAP_ORCAMENTO <<< "$BUDGET_CALC"

log "total_timesteps escolhido: ${TOTAL_TIMESTEPS}"
log "motivo: min(3x linhas_train=${CAP_EPOCAS}, passos_por_segundo x ORCAMENTO_SEG(${ORCAMENTO_SEG}s)=${CAP_ORCAMENTO})"

if [ "$TOTAL_TIMESTEPS" -lt "$N_LINHAS_TRAIN" ]; then
    log "############################################################"
    log "# WARNING: total_timesteps (${TOTAL_TIMESTEPS}) < 1 passada completa pelos"
    log "# dados de treino (${N_LINHAS_TRAIN} linhas). O orçamento de tempo"
    log "# (ORCAMENTO_SEG=${ORCAMENTO_SEG}s) é pequeno demais para essa granularidade."
    log "# RECOMENDADO: rode com --resample 15min (menos linhas) ou use a VPS da Fase 6"
    log "# (mais CPU/tempo). Prosseguindo mesmo assim, conforme solicitado."
    log "############################################################"
fi

log "== (e) Treinando PPO por ${TOTAL_TIMESTEPS} passos (checkpoint a cada 25k, resume automático, N_ENVS=${N_ENVS}) =="
python scripts/train.py --train train_data.csv --total-timesteps "$TOTAL_TIMESTEPS"

log "== (f) Avaliando em val_data.csv =="
python scripts/evaluate.py --trade val_data.csv --agents ppo

log "== (g) Backtest 8.5: signals.py (contrato date->weight) + backtest_pro.py (tearsheet) =="
python scripts/signals.py --trade trade_data.csv --agent ppo --out "${RESULTS_DIR}/weights_ppo.csv"
python scripts/backtest_pro.py --trade trade_data.csv --weights "${RESULTS_DIR}/weights_ppo.csv" \
    --fee 0.001 --to-daily --out "${RESULTS_DIR}/tearsheet.html"

END_TS=$(python -c 'import time; print(time.time())')
WALL_SECONDS=$(python -c "print(int(${END_TS} - ${START_TS}))")

log "== (h) RESUMO FINAL =="
log "total_timesteps usado: ${TOTAL_TIMESTEPS}"
log "tempo de parede total: ${WALL_SECONDS}s"
log "métricas de validação (val_data.csv): ver bloco '(f)' acima"
log "métricas de trade (trade_data.csv): ver bloco '(g)' acima"
log "tearsheet: ${RESULTS_DIR}/tearsheet.html"
log "modelo final: trained_models/agent_ppo.zip"
log "Concluído."
