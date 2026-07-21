"""Fase 8.8 — utilitários de agregação do ensemble. Zero import de FinRL:
opera só sobre os CSVs produzidos por signals.py/backtest_pro.py (contrato
date->weight e métricas), na linha do "anti-Frankenstein" da Fase 8.5.

Subcomandos:
  committee   -- média simples dos pesos (date->weight) de N braços
  winner      -- escolhe o braço com melhor Sharpe de validação, ignorando
                 braços de risco (lambda>0) colapsados (peso ~0, retorno 0,
                 Sharpe NaN); imprime só o rótulo (stdout)
  peso-medio  -- imprime o peso médio (mean, não abs) de um CSV date->weight
  summary     -- concatena métricas de trade de vários braços (+ baselines:
                 buy_and_hold, momentum, comite) num único results/summary.csv,
                 com peso_medio_trade, dsr e prob_sharpe>0 (Fase 8.9, passo 4)
  aggregate   -- lê summary.csv e agrega por algoritmo (média±desvio do Sharpe
                 e do retorno entre sementes) — Fase 8.9, passo 5
"""
from __future__ import annotations

import argparse
import sys
from math import e as euler_e
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

# Abaixo desse limiar, retorno_acumulado/peso_medio são tratados como "zero"
# (braço colapsado: parou de operar). Sharpe NaN sozinho já basta.
COLLAPSE_EPS = 1e-6

# Rótulos reservados que NÃO são "tentativas" de busca de hiperparâmetros
# (são baselines/derivados) -- excluídos do cálculo de SR0 (Fase 8.9, passo 4)
# e do agrupamento por algoritmo (passo 5).
BASELINE_LABELS = {"buy_and_hold", "momentum", "comite"}

EULER_MASCHERONI = 0.5772156649015329  # γ


def expected_max_sharpe(n_trials: int, sharpe_estimates: list[float]) -> float:
    """SR0*: Sharpe esperado sob H0 ao escolher o melhor dentre `n_trials`
    tentativas independentes (Bailey & López de Prado 2014, eq. da teoria de
    valores extremos), usando a variância cross-sectional dos Sharpes
    observados nesta rodada como proxy de Var[SR_n]. Com <2 tentativas válidas
    (sem dispersão para estimar), devolve 0.0 -- degenera para PSR(0)."""
    valid = [s for s in sharpe_estimates if pd.notna(s)]
    if n_trials <= 1 or len(valid) < 2:
        return 0.0
    var_sr = float(np.var(valid, ddof=1))
    if var_sr <= 0:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * euler_e))
    return sqrt(var_sr) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def probabilistic_sharpe_ratio(sharpe_bruto: float, sharpe_threshold: float, n_obs: int,
                                skew: float, kurtosis_pearson: float) -> float:
    """PSR(SR*) (Bailey & López de Prado 2014): probabilidade de o Sharpe
    (NÃO anualizado) observado exceder `sharpe_threshold`, corrigindo pelo nº
    de observações e pela não-normalidade (assimetria/curtose) dos retornos.
    Com <2 observações ou denominador não-positivo, devolve NaN (dado
    insuficiente para a estimativa)."""
    if pd.isna(sharpe_bruto) or n_obs <= 1:
        return float("nan")
    denom = 1 - skew * sharpe_bruto + ((kurtosis_pearson - 1) / 4.0) * sharpe_bruto**2
    if denom <= 0:
        return float("nan")
    z = (sharpe_bruto - sharpe_threshold) * sqrt(n_obs - 1) / sqrt(denom)
    return float(norm.cdf(z))


def mean_weight(weights_path: str) -> float:
    """Peso médio (mean, não abs) de um contrato date->weight -- mesmo cálculo
    que signals.py já imprime (w.weight.mean())."""
    return float(pd.read_csv(weights_path)["weight"].mean())


def build_committee_weights(weight_paths: list[str]) -> pd.DataFrame:
    """Comitê (Fase 8.8) = média simples do peso entre os braços informados,
    alinhada por data (união de datas; braço ausente numa data não participa
    da média nessa barra)."""
    series = [pd.read_csv(p).set_index("date")["weight"] for p in weight_paths]
    combined = pd.concat(series, axis=1)
    avg = combined.mean(axis=1)
    return pd.DataFrame({"date": avg.index, "weight": avg.values})


def _is_collapsed(metrics_path: str, weights_path: str) -> tuple[bool, float]:
    """Braço colapsado = env parou de operar: Sharpe NaN, ou retorno_acumulado
    e peso médio ambos ~0. Retorna (colapsado, sharpe) para diagnóstico."""
    df = pd.read_csv(metrics_path)
    row = df.loc[df["label"] == "estrategia"].iloc[0]
    sharpe = float(row["sharpe_aproximado"])
    retorno = float(row["retorno_acumulado"])
    peso_medio = mean_weight(weights_path)
    collapsed = pd.isna(sharpe) or (abs(retorno) < COLLAPSE_EPS and abs(peso_medio) < COLLAPSE_EPS)
    return collapsed, sharpe


def pick_winner(branches: list[tuple[str, str, str, str]]) -> str:
    """Escolhe o braço (rótulo) com melhor Sharpe de validação, ignorando
    braços de risco (lambda>0) colapsados. Se TODOS os braços de risco
    estiverem colapsados, cai no braço de controle (lambda==0).

    branches: [(rotulo, lambda_str, metrics_path, weights_path), ...] (val).

    Diagnóstico vai para stderr — stdout fica só com o rótulo vencedor, para o
    orquestrador (run_ensemble.sh) capturar com $(...) sem ruído.
    """
    candidates: list[tuple[str, float]] = []
    control_label = None
    any_risk_alive = False

    for label, lambda_str, metrics_path, weights_path in branches:
        collapsed, sharpe = _is_collapsed(metrics_path, weights_path)
        is_control = float(lambda_str) <= 0
        status = "colapsou (ignorado)" if collapsed else f"sharpe={sharpe:.4f}"
        print(f"val {label} (lambda={lambda_str}): {status}", file=sys.stderr)

        if is_control:
            control_label = label
            if pd.notna(sharpe):
                candidates.append((label, sharpe))
            continue

        if collapsed:
            continue
        any_risk_alive = True
        if pd.notna(sharpe):
            candidates.append((label, sharpe))

    if not any_risk_alive:
        if control_label is None:
            raise ValueError("nenhum braço com Sharpe válido (todos NaN ou colapsados?) — não há vencedor")
        print("aviso: todos os braços com risco colapsaram — vencedor é o controle", file=sys.stderr)
        return control_label

    if not candidates:
        raise ValueError("nenhum braço com Sharpe válido (todos NaN ou colapsados?) — não há vencedor")

    best_label, _ = max(candidates, key=lambda item: item[1])
    return best_label


def build_summary(
    labeled_metrics_paths: list[tuple[str, str]],
    weights_by_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Concatena a linha 'estrategia' (renomeada com o rótulo do braço) de cada
    CSV de métricas de trade, mais UMA linha 'buy_and_hold' (idêntica entre
    braços, pois é o mesmo benchmark — pega a do primeiro arquivo). Adiciona a
    coluna 'peso_medio_trade' (mean, não abs) a partir do weights de trade de
    cada braço -- NaN para buy_and_hold (sem weights próprio) e para rótulos
    sem entrada em weights_by_label. Também adiciona 'dsr' e 'prob_sharpe>0'
    (Fase 8.9, passo 4) via _add_dsr_columns."""
    weights_by_label = weights_by_label or {}
    rows = []
    bh_added = False
    for label, path in labeled_metrics_paths:
        df = pd.read_csv(path)
        strat = df.loc[df["label"] == "estrategia"].iloc[0].to_dict()
        strat["label"] = label
        w_path = weights_by_label.get(label)
        strat["peso_medio_trade"] = mean_weight(w_path) if w_path else float("nan")
        rows.append(strat)
        if not bh_added:
            bh = df.loc[df["label"] == "buy_and_hold"].iloc[0].to_dict()
            bh["peso_medio_trade"] = float("nan")
            rows.append(bh)
            bh_added = True
    summary = pd.DataFrame(rows)
    _add_dsr_columns(summary)
    return summary


def _add_dsr_columns(summary: pd.DataFrame) -> None:
    """Adiciona, in-place, as colunas 'dsr' e 'prob_sharpe>0' (Fase 8.9, passo
    4 -- Bailey & López de Prado 2014). SR0 (Sharpe esperado por acaso ao
    escolher o melhor dentre várias tentativas) é estimado a partir da
    dispersão de sharpe_bruto entre as tentativas de busca desta rodada
    (rótulos fora de BASELINE_LABELS); N = nº dessas tentativas ("número de
    braços desta rodada"). dsr = PSR(SR0) (corrige Sharpe>SR0 por múltiplas
    comparações); prob_sharpe>0 = PSR(0) (só corrige por amostra finita e
    não-normalidade, sem a correção de múltiplas comparações)."""
    if "sharpe_bruto" not in summary.columns:
        summary["dsr"] = float("nan")
        summary["prob_sharpe>0"] = float("nan")
        return

    trial_mask = ~summary["label"].isin(BASELINE_LABELS)
    n_trials = int(trial_mask.sum())
    sr0 = expected_max_sharpe(n_trials, summary.loc[trial_mask, "sharpe_bruto"].tolist())
    print(f"DSR: n_trials={n_trials}  SR0(esperado por acaso)={sr0:.4f}", file=sys.stderr)

    dsr_vals, psr0_vals = [], []
    for _, row in summary.iterrows():
        sharpe_bruto = row.get("sharpe_bruto", float("nan"))
        n_obs_raw = row.get("n_obs", float("nan"))
        n_obs = int(n_obs_raw) if pd.notna(n_obs_raw) else 0
        skew = row.get("assimetria", 0.0)
        kurt = row.get("curtose_pearson", 3.0)
        dsr_vals.append(probabilistic_sharpe_ratio(sharpe_bruto, sr0, n_obs, skew, kurt))
        psr0_vals.append(probabilistic_sharpe_ratio(sharpe_bruto, 0.0, n_obs, skew, kurt))
    summary["dsr"] = dsr_vals
    summary["prob_sharpe>0"] = psr0_vals


def aggregate_by_algo(summary: pd.DataFrame) -> pd.DataFrame:
    """Fase 8.9, passo 5: agrega o summary por algoritmo (extraído do rótulo
    '<algo>_l<lambda>_s<seed>' -> prefixo antes de '_l'), reportando
    média±desvio do Sharpe e do retorno entre as sementes -- para julgar
    robustez/dispersão do algoritmo, não só o braço vencedor pontual."""
    trial = summary.loc[~summary["label"].isin(BASELINE_LABELS)].copy()
    trial["algo"] = trial["label"].str.split("_l", n=1).str[0]
    return trial.groupby("algo").agg(
        n_sementes=("label", "count"),
        sharpe_medio=("sharpe_aproximado", "mean"),
        sharpe_desvio=("sharpe_aproximado", "std"),
        retorno_medio=("retorno_acumulado", "mean"),
        retorno_desvio=("retorno_acumulado", "std"),
    ).reset_index()


def _parse_labeled(items: list[str], n_fields: int = 2) -> list[tuple[str, ...]]:
    """Converte 'campo1=campo2=...=campoN' repetido em [(campo1, ..., campoN), ...].

    n_fields=2: 'rotulo=caminho.csv' (usado por committee/summary legado).
    n_fields=3: 'rotulo=metrics.csv=weights.csv' (summary).
    n_fields=4: 'rotulo=lambda=metrics.csv=weights.csv' (winner).
    """
    out = []
    for item in items:
        parts = item.split("=", n_fields - 1)
        if len(parts) != n_fields:
            raise ValueError(f"esperado {n_fields} campos separados por '=', recebido: {item!r}")
        out.append(tuple(parts))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_committee = sub.add_parser("committee", help="média dos pesos de N braços")
    p_committee.add_argument("--weights", nargs="+", required=True, help="CSVs date,weight")
    p_committee.add_argument("--out", required=True)

    p_winner = sub.add_parser("winner", help="escolhe o braço com melhor Sharpe de validação (robusto a colapso)")
    p_winner.add_argument("--branches", nargs="+", required=True,
                           help="'rotulo=lambda=metrics.csv=weights.csv' (val), um por braço")

    p_peso = sub.add_parser("peso-medio", help="imprime o peso médio (mean, não abs) de um CSV date->weight")
    p_peso.add_argument("--weights", required=True)

    p_summary = sub.add_parser("summary", help="monta o summary.csv de comparação (trade)")
    p_summary.add_argument("--branches", nargs="+", required=True,
                            help="'rotulo=metrics.csv=weights.csv' (trade), um por braço")
    p_summary.add_argument("--out", required=True)

    p_aggregate = sub.add_parser("aggregate", help="agrega o summary.csv por algoritmo (média±desvio entre sementes)")
    p_aggregate.add_argument("--summary", required=True)
    p_aggregate.add_argument("--out", required=True)

    args = p.parse_args()

    if args.cmd == "committee":
        build_committee_weights(args.weights).to_csv(args.out, index=False)
        print(f"pesos do comitê salvos em {args.out}")
    elif args.cmd == "winner":
        branches = _parse_labeled(args.branches, n_fields=4)
        print(pick_winner(branches))
    elif args.cmd == "peso-medio":
        print(f"{mean_weight(args.weights):.6f}")
    elif args.cmd == "summary":
        parsed = _parse_labeled(args.branches, n_fields=3)
        labeled_metrics = [(label, metrics_path) for label, metrics_path, _ in parsed]
        weights_by_label = {label: weights_path for label, _, weights_path in parsed}
        summary = build_summary(labeled_metrics, weights_by_label)
        summary.to_csv(args.out, index=False)
        print(f"summary salvo em {args.out}")
        print(summary.to_string(index=False))
    elif args.cmd == "aggregate":
        summary = pd.read_csv(args.summary)
        agg = aggregate_by_algo(summary)
        agg.to_csv(args.out, index=False)
        print(f"agregado por algoritmo salvo em {args.out}")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
