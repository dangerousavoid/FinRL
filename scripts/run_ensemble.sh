#!/usr/bin/env bash
# Fase 8.8 — orquestrador do ensemble de recompensa com risco.
#
# Roda em sequência os braços:
#   ppo_l0_s1     (controle: PPO, RISK_LAMBDA=0, seed=1 — reproduz o env base)
#   ppo_l0.5_s1   (PPO, RISK_LAMBDA=0.5, seed=1)
#   ppo_l0.5_s2   (PPO, RISK_LAMBDA=0.5, seed=2)
#   a2c_l0.5_s1   (A2C, RISK_LAMBDA=0.5, seed=1)
#   sac_l0.5_s1   (SAC, RISK_LAMBDA=0.5, seed=1)
#
# Para cada braço: treina (scripts/train.py), extrai o contrato date->weight
# em val e trade (scripts/signals.py, Fase 8.5) e roda o backtest reamostrado
# p/ diária (scripts/backtest_pro.py --to-daily). Ao final: escolhe o braço de
# melhor Sharpe de VALIDAÇÃO (tearsheet dele no trade), monta o comitê (média
# dos pesos dos braços com LAMBDA>0) e escreve results/summary.csv comparando
# todos os braços + comitê + buy-and-hold no trade.
#
# Variáveis de ambiente:
#   RESAMPLE            granularidade do adaptador (default: 1h, Fase 8.8)
#   CSV_DIR             pasta com os CSVs brutos da CDD (default: data/raw)
#   N_ENVS              nº de envs paralelos por braço (default: 1)
#   FRESH               1 = esvazia trained_models/ e results/ antes do 1º
#                       braço (preserva run.log); 0 (default) = preserva p/
#                       permitir resume de um ensemble interrompido
#   TIMESTEPS_OVERRIDE  se definida, sobrepõe --total-timesteps em TODOS os
#                       braços (útil só p/ smoke test da orquestração — ver
#                       Fase 8.8 do plano); vazio (default) = cada braço usa o
#                       default de train.py (PASSADAS x linhas do train)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESAMPLE="${RESAMPLE:-1h}"
CSV_DIR="${CSV_DIR:-data/raw}"
N_ENVS="${N_ENVS:-1}"
FRESH="${FRESH:-0}"
TIMESTEPS_OVERRIDE="${TIMESTEPS_OVERRIDE:-}"
RESULTS_DIR="results"

export N_ENVS

log() { echo "[run_ensemble] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }

if [ "$FRESH" = "1" ]; then
    log "== (a) Começo limpo (FRESH=1): esvaziando trained_models/ e results/ (preservando run.log) =="
    find trained_models -mindepth 1 -delete 2>/dev/null || true
    find "$RESULTS_DIR" -mindepth 1 ! -name 'run.log' -delete 2>/dev/null || true
fi
mkdir -p trained_models "$RESULTS_DIR"

log "== (b) Gerando datasets (resample=${RESAMPLE}) a partir de ${CSV_DIR} =="
python scripts/cdd_to_finrl.py --csv "$CSV_DIR" --tic BTCUSDT --resample "$RESAMPLE" \
    --train-out train_data.csv --val-out val_data.csv --trade-out trade_data.csv

# tag algo lambda seed
BRANCHES=(
    "ppo_l0_s1 ppo 0 1"
    "ppo_l0.5_s1 ppo 0.5 1"
    "ppo_l0.5_s2 ppo 0.5 2"
    "a2c_l0.5_s1 a2c 0.5 1"
    "sac_l0.5_s1 sac 0.5 1"
)

VAL_METRICS_ARGS=()
TRADE_METRICS_ARGS=()
LAMBDA_POS_TRADE_WEIGHTS=()

N_BRANCHES=${#BRANCHES[@]}
i=0
for branch in "${BRANCHES[@]}"; do
    i=$((i + 1))
    read -r TAG ALGO LAMBDA SEED <<< "$branch"
    log "############################################################"
    log "== Braço ${i}/${N_BRANCHES}: ${TAG}  (algo=${ALGO} RISK_LAMBDA=${LAMBDA} seed=${SEED}) =="
    log "############################################################"

    export RISK_LAMBDA="$LAMBDA"

    TIMESTEPS_ARGS=()
    if [ -n "$TIMESTEPS_OVERRIDE" ]; then
        TIMESTEPS_ARGS=(--total-timesteps "$TIMESTEPS_OVERRIDE")
    fi

    log "-- (1) Treinando ${TAG} --"
    python scripts/train.py --train train_data.csv --algo "$ALGO" --seed "$SEED" \
        --checkpoint-prefix "checkpoint_${TAG}" --model-name "agent_${TAG}" \
        "${TIMESTEPS_ARGS[@]}"

    for split in val trade; do
        SPLIT_CSV="${split}_data.csv"
        WEIGHTS_OUT="${RESULTS_DIR}/weights_${TAG}_${split}.csv"
        METRICS_OUT="${RESULTS_DIR}/metrics_${TAG}_${split}.csv"
        TEARSHEET_OUT="${RESULTS_DIR}/tearsheet_${TAG}_${split}.html"

        log "-- (2) Contrato date->weight + backtest reamostrado p/ diária (${split}) --"
        python scripts/signals.py --trade "$SPLIT_CSV" --agent "$ALGO" \
            --model-path "trained_models/agent_${TAG}" --out "$WEIGHTS_OUT"
        python scripts/backtest_pro.py --trade "$SPLIT_CSV" --weights "$WEIGHTS_OUT" \
            --fee 0.001 --to-daily --out "$TEARSHEET_OUT" --metrics-out "$METRICS_OUT"

        if [ "$split" = "val" ]; then
            VAL_METRICS_ARGS+=("${TAG}=${METRICS_OUT}")
        else
            TRADE_METRICS_ARGS+=("${TAG}=${METRICS_OUT}")
            # LAMBDA=0 é o controle/diagnóstico — comitê é só dos braços com LAMBDA>0
            if [ "$LAMBDA" != "0" ]; then
                LAMBDA_POS_TRADE_WEIGHTS+=("$WEIGHTS_OUT")
            fi
        fi
    done
done

log "############################################################"
log "== (c) Selecionando o braço de melhor Sharpe de VALIDAÇÃO =="
log "############################################################"
WINNER=$(python scripts/ensemble_select.py winner --metrics "${VAL_METRICS_ARGS[@]}")
log "vencedor (Sharpe de validação): ${WINNER}"
cp "${RESULTS_DIR}/tearsheet_${WINNER}_trade.html" "${RESULTS_DIR}/tearsheet_vencedor.html"
log "tearsheet do vencedor (no trade): ${RESULTS_DIR}/tearsheet_vencedor.html"

log "############################################################"
log "== (d) Comitê: média dos pesos dos braços com LAMBDA>0 (no trade) =="
log "############################################################"
COMITE_WEIGHTS="${RESULTS_DIR}/weights_comite_trade.csv"
COMITE_METRICS="${RESULTS_DIR}/metrics_comite_trade.csv"
python scripts/ensemble_select.py committee --weights "${LAMBDA_POS_TRADE_WEIGHTS[@]}" --out "$COMITE_WEIGHTS"
python scripts/backtest_pro.py --trade trade_data.csv --weights "$COMITE_WEIGHTS" \
    --fee 0.001 --to-daily --out "${RESULTS_DIR}/tearsheet_comite.html" --metrics-out "$COMITE_METRICS"
log "tearsheet do comitê (no trade): ${RESULTS_DIR}/tearsheet_comite.html"

log "############################################################"
log "== (e) results/summary.csv: todos os braços + comitê + buy-and-hold (no trade) =="
log "############################################################"
python scripts/ensemble_select.py summary \
    --metrics "${TRADE_METRICS_ARGS[@]}" "comite=${COMITE_METRICS}" \
    --out "${RESULTS_DIR}/summary.csv"

log "== Concluído. Ver ${RESULTS_DIR}/summary.csv, ${RESULTS_DIR}/tearsheet_vencedor.html e ${RESULTS_DIR}/tearsheet_comite.html =="
