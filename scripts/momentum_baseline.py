"""Fase 8.9, passo 3 — baseline de momentum simples (novo, aditivo).

Comprado (peso 1) quando close > média móvel de N barras; fora (peso 0) caso
contrário. Produz o MESMO contrato date->weight de scripts/signals.py, então
consome os MESMOS custos de turnover em scripts/backtest_pro.py (--fee) que
qualquer braço do ensemble -- comparação justa, sem tratamento especial.

Janela via env var MOM_WINDOW (default 168 barras = ~1 semana em barras de 1h).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def momentum_weights(trade: pd.DataFrame, window: int) -> pd.DataFrame:
    prices = trade.groupby("date")["close"].first().sort_index()
    sma = prices.rolling(window=window, min_periods=window).mean()
    # warm-up (sma ainda NaN): sem sinal -> fora (peso 0), nunca opera "no escuro"
    weight = (prices > sma).astype(float).where(sma.notna(), 0.0)
    return pd.DataFrame({"date": prices.index, "weight": weight.values})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--window", type=int, default=int(os.environ.get("MOM_WINDOW", "168")),
                   help="nº de barras da média móvel (default: env var MOM_WINDOW ou 168)")
    p.add_argument("--out", default="results/weights_momentum.csv")
    args = p.parse_args()

    trade = pd.read_csv(args.trade)
    w = momentum_weights(trade, window=args.window)
    w.to_csv(args.out, index=False)
    print(f"pesos de momentum salvos em {args.out} (janela={args.window} barras, peso médio={w.weight.mean():.3f})")


if __name__ == "__main__":
    main()
