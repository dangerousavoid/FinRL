#!/usr/bin/env bash
# Fase 8.8 — orquestrador do ensemble de recompensa com risco / varredura de
# hiperparâmetros.
#
# Os braços são o produto cartesiano ALGOS x LAMBDA_GRID x SEEDS, com uma
# exceção: LAMBDA=0 é o controle/diagnóstico (reproduz o env original) e só
# faz sentido rodar UMA vez, então só é gerado para o primeiro algo de ALGOS
# e a primeira seed de SEEDS (evita repetir o mesmo controle à toa).
# Com os defaults (LAMBDA_GRID="0 0.5", ALGOS="ppo a2c sac", SEEDS="1") isso
# gera 4 braços: ppo_l0_s1 (controle), ppo_l0.5_s1, a2c_l0.5_s1, sac_l0.5_s1.
# (Note: são 4, não os 5 de antes — a 2ª seed do PPO saiu do hardcode; para
# rodar 2 seeds em todo mundo use SEEDS="1 2".)
#
# Para cada braço: treina (scripts/train.py), extrai o contrato date->weight
# em val e trade (scripts/signals.py, Fase 8.5) e roda o backtest reamostrado
# p/ diária (scripts/backtest_pro.py --to-daily). Ao final: escolhe o braço de
# melhor Sharpe de VALIDAÇÃO (tearsheet dele no trade), ignorando braços de
# risco (lambda>0) colapsados (peso ~0, retorno 0, Sharpe NaN) — se todos os
# braços de risco colapsarem, o vencedor cai no controle. Monta o comitê
# (média dos pesos dos braços com LAMBDA>0 que NÃO colapsaram no trade; se
# nenhum sobrar, pula o comitê). Escreve results/summary.csv comparando todos
# os braços + comitê (se houver) + buy-and-hold no trade, com peso_medio_trade
# por braço.
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
#   LAMBDA_GRID         lista de RISK_LAMBDA separada por espaço p/ varrer
#                       (default: "0 0.5")
#   ALGOS               lista de algoritmos separada por espaço: ppo/a2c/sac
#                       (default: "ppo a2c sac")
#   SEEDS               lista de seeds separada por espaço (default: "1")
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESAMPLE="${RESAMPLE:-1h}"
CSV_DIR="${CSV_DIR:-data/raw}"
N_ENVS="${N_ENVS:-1}"
FRESH="${FRESH:-0}"
TIMESTEPS_OVERRIDE="${TIMESTEPS_OVERRIDE:-}"
LAMBDA_GRID="${LAMBDA_GRID:-0 0.5}"
ALGOS="${ALGOS:-ppo a2c sac}"
SEEDS="${SEEDS:-1}"
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

# ---- Montagem dos braços: produto cartesiano ALGOS x LAMBDA_GRID x SEEDS ----
# tag algo lambda seed
read -r -a ALGOS_ARR <<< "$ALGOS"
read -r -a LAMBDA_ARR <<< "$LAMBDA_GRID"
read -r -a SEEDS_ARR <<< "$SEEDS"
FIRST_ALGO="${ALGOS_ARR[0]}"
FIRST_SEED="${SEEDS_ARR[0]}"

BRANCHES=()
for ALGO in "${ALGOS_ARR[@]}"; do
    for LAMBDA in "${LAMBDA_ARR[@]}"; do
        for SEED in "${SEEDS_ARR[@]}"; do
            if [ "$LAMBDA" = "0" ]; then
                # controle: só uma vez (primeiro algo, primeira seed)
                if [ "$ALGO" != "$FIRST_ALGO" ] || [ "$SEED" != "$FIRST_SEED" ]; then
                    continue
                fi
            fi
            TAG="${ALGO}_l${LAMBDA}_s${SEED}"
            BRANCHES+=("${TAG} ${ALGO} ${LAMBDA} ${SEED}")
        done
    done
done

log "== Braços do ensemble (${#BRANCHES[@]} no total) =="
for branch in "${BRANCHES[@]}"; do
    read -r TAG ALGO LAMBDA SEED <<< "$branch"
    log "  - ${TAG}  (algo=${ALGO} RISK_LAMBDA=${LAMBDA} seed=${SEED})"
done

VAL_BRANCHES_ARGS=()
TRADE_SUMMARY_ARGS=()
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
            VAL_BRANCHES_ARGS+=("${TAG}=${LAMBDA}=${METRICS_OUT}=${WEIGHTS_OUT}")
        else
            TRADE_SUMMARY_ARGS+=("${TAG}=${METRICS_OUT}=${WEIGHTS_OUT}")
            # comitê: só braços com risco (lambda>0) e que não colapsaram (peso médio > 0)
            if [ "$LAMBDA" != "0" ]; then
                PESO_MEDIO=$(python scripts/ensemble_select.py peso-medio --weights "$WEIGHTS_OUT")
                if awk -v p="$PESO_MEDIO" 'BEGIN { p = (p < 0 ? -p : p); exit !(p > 1e-6) }'; then
                    LAMBDA_POS_TRADE_WEIGHTS+=("$WEIGHTS_OUT")
                else
                    log "aviso: ${TAG} colapsou no trade (peso médio=${PESO_MEDIO}) — excluído do comitê"
                fi
            fi
        fi
    done
done

log "############################################################"
log "== (c) Selecionando o braço de melhor Sharpe de VALIDAÇÃO (robusto a colapso) =="
log "############################################################"
WINNER=$(python scripts/ensemble_select.py winner --branches "${VAL_BRANCHES_ARGS[@]}")
log "vencedor (Sharpe de validação): ${WINNER}"
TEARSHEET_VENCEDOR_FEITO=0
if [ -f "${RESULTS_DIR}/tearsheet_${WINNER}_trade.html" ]; then
    cp "${RESULTS_DIR}/tearsheet_${WINNER}_trade.html" "${RESULTS_DIR}/tearsheet_vencedor.html"
    log "tearsheet do vencedor (no trade): ${RESULTS_DIR}/tearsheet_vencedor.html"
    TEARSHEET_VENCEDOR_FEITO=1
else
    log "aviso: backtest_pro.py não gerou tearsheet HTML p/ ${WINNER} (provável colapso/variância zero) — sem tearsheet_vencedor.html"
fi

log "############################################################"
log "== (d) Comitê: média dos pesos dos braços com LAMBDA>0 que não colapsaram (no trade) =="
log "############################################################"
COMITE_WEIGHTS="${RESULTS_DIR}/weights_comite_trade.csv"
COMITE_METRICS="${RESULTS_DIR}/metrics_comite_trade.csv"
COMITE_FEITO=0
if [ "${#LAMBDA_POS_TRADE_WEIGHTS[@]}" -eq 0 ]; then
    log "aviso: nenhum braço com risco sobrou para o comitê (todos colapsaram) — pulando comitê"
else
    python scripts/ensemble_select.py committee --weights "${LAMBDA_POS_TRADE_WEIGHTS[@]}" --out "$COMITE_WEIGHTS"
    python scripts/backtest_pro.py --trade trade_data.csv --weights "$COMITE_WEIGHTS" \
        --fee 0.001 --to-daily --out "${RESULTS_DIR}/tearsheet_comite.html" --metrics-out "$COMITE_METRICS"
    log "tearsheet do comitê (no trade): ${RESULTS_DIR}/tearsheet_comite.html"
    COMITE_FEITO=1
fi

log "############################################################"
log "== (e) results/summary.csv: todos os braços + comitê (se houver) + buy-and-hold (no trade) =="
log "############################################################"
SUMMARY_BRANCHES_ARGS=("${TRADE_SUMMARY_ARGS[@]}")
if [ "$COMITE_FEITO" -eq 1 ]; then
    SUMMARY_BRANCHES_ARGS+=("comite=${COMITE_METRICS}=${COMITE_WEIGHTS}")
fi
python scripts/ensemble_select.py summary \
    --branches "${SUMMARY_BRANCHES_ARGS[@]}" \
    --out "${RESULTS_DIR}/summary.csv"

log "== Concluído. Ver ${RESULTS_DIR}/summary.csv =="
if [ "$TEARSHEET_VENCEDOR_FEITO" -eq 1 ]; then
    log "== e ${RESULTS_DIR}/tearsheet_vencedor.html =="
fi
if [ "$COMITE_FEITO" -eq 1 ]; then
    log "== e ${RESULTS_DIR}/tearsheet_comite.html =="
fi
