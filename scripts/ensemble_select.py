"""Fase 8.8 — utilitários de agregação do ensemble. Zero import de FinRL:
opera só sobre os CSVs produzidos por signals.py/backtest_pro.py (contrato
date->weight e métricas), na linha do "anti-Frankenstein" da Fase 8.5.

Subcomandos:
  committee   -- média simples dos pesos (date->weight) de N braços
  winner      -- escolhe o braço com melhor Sharpe de validação, ignorando
                 braços de risco (lambda>0) colapsados (peso ~0, retorno 0,
                 Sharpe NaN); imprime só o rótulo (stdout)
  peso-medio  -- imprime o peso médio (mean, não abs) de um CSV date->weight
  summary     -- concatena métricas de trade de vários braços (+ 1 buy_and_hold)
                 num único results/summary.csv, com a coluna peso_medio_trade
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

# Abaixo desse limiar, retorno_acumulado/peso_medio são tratados como "zero"
# (braço colapsado: parou de operar). Sharpe NaN sozinho já basta.
COLLAPSE_EPS = 1e-6


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
    sem entrada em weights_by_label."""
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
    return pd.DataFrame(rows)


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


if __name__ == "__main__":
    main()
