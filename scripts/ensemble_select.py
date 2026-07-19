"""Fase 8.8 — utilitários de agregação do ensemble. Zero import de FinRL:
opera só sobre os CSVs produzidos por signals.py/backtest_pro.py (contrato
date->weight e métricas), na linha do "anti-Frankenstein" da Fase 8.5.

Subcomandos:
  committee  -- média simples dos pesos (date->weight) de N braços
  winner     -- escolhe o braço com melhor Sharpe (linha 'estrategia') numa
                tabela de métricas de validação; imprime só o rótulo (stdout)
  summary    -- concatena métricas de trade de vários braços (+ 1 buy_and_hold)
                num único results/summary.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def build_committee_weights(weight_paths: list[str]) -> pd.DataFrame:
    """Comitê (Fase 8.8) = média simples do peso entre os braços informados,
    alinhada por data (união de datas; braço ausente numa data não participa
    da média nessa barra)."""
    series = [pd.read_csv(p).set_index("date")["weight"] for p in weight_paths]
    combined = pd.concat(series, axis=1)
    avg = combined.mean(axis=1)
    return pd.DataFrame({"date": avg.index, "weight": avg.values})


def pick_winner(labeled_metrics_paths: list[tuple[str, str]]) -> str:
    """Escolhe o braço (rótulo) com melhor Sharpe de validação (linha 'estrategia').

    Diagnóstico vai para stderr — stdout fica só com o rótulo vencedor, para o
    orquestrador (run_ensemble.sh) capturar com $(...) sem ruído.
    """
    best_label, best_sharpe = None, float("-inf")
    for label, path in labeled_metrics_paths:
        df = pd.read_csv(path)
        sharpe = float(df.loc[df["label"] == "estrategia", "sharpe_aproximado"].iloc[0])
        print(f"val Sharpe {label}: {sharpe:.4f}", file=sys.stderr)
        if pd.notna(sharpe) and sharpe > best_sharpe:
            best_label, best_sharpe = label, sharpe
    if best_label is None:
        raise ValueError("nenhum braço com Sharpe válido (todos NaN?) — não há vencedor")
    return best_label


def build_summary(labeled_metrics_paths: list[tuple[str, str]]) -> pd.DataFrame:
    """Concatena a linha 'estrategia' (renomeada com o rótulo do braço) de cada
    CSV de métricas de trade, mais UMA linha 'buy_and_hold' (idêntica entre
    braços, pois é o mesmo benchmark — pega a do primeiro arquivo)."""
    rows = []
    bh_added = False
    for label, path in labeled_metrics_paths:
        df = pd.read_csv(path)
        strat = df.loc[df["label"] == "estrategia"].iloc[0].to_dict()
        strat["label"] = label
        rows.append(strat)
        if not bh_added:
            bh = df.loc[df["label"] == "buy_and_hold"].iloc[0].to_dict()
            rows.append(bh)
            bh_added = True
    return pd.DataFrame(rows)


def _parse_labeled(items: list[str]) -> list[tuple[str, str]]:
    """Converte 'rotulo=caminho.csv' repetido em [(rotulo, caminho), ...]."""
    out = []
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError(f"esperado 'rotulo=caminho.csv', recebido: {item!r}")
        out.append((label, path))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_committee = sub.add_parser("committee", help="média dos pesos de N braços")
    p_committee.add_argument("--weights", nargs="+", required=True, help="CSVs date,weight")
    p_committee.add_argument("--out", required=True)

    p_winner = sub.add_parser("winner", help="escolhe o braço com melhor Sharpe de validação")
    p_winner.add_argument("--metrics", nargs="+", required=True, help="'rotulo=caminho.csv' (val)")

    p_summary = sub.add_parser("summary", help="monta o summary.csv de comparação (trade)")
    p_summary.add_argument("--metrics", nargs="+", required=True, help="'rotulo=caminho.csv' (trade)")
    p_summary.add_argument("--out", required=True)

    args = p.parse_args()

    if args.cmd == "committee":
        build_committee_weights(args.weights).to_csv(args.out, index=False)
        print(f"pesos do comitê salvos em {args.out}")
    elif args.cmd == "winner":
        # única linha em stdout: o orquestrador captura com $(...)
        print(pick_winner(_parse_labeled(args.metrics)))
    elif args.cmd == "summary":
        summary = build_summary(_parse_labeled(args.metrics))
        summary.to_csv(args.out, index=False)
        print(f"summary salvo em {args.out}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
