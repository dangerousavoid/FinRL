from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import quantstats as qs


def run_backtest(
    prices: pd.Series, weights: pd.Series, fee: float = 0.001, to_daily: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Recebe {preços, pesos} — ZERO import de FinRL/FinRL-X. Aplica o peso de
    forma causal (sem look-ahead) e desconta custo proporcional ao turnover."""
    idx = prices.index.intersection(weights.index)
    prices, weights = prices.loc[idx].sort_index(), weights.loc[idx].sort_index()

    ret = prices.pct_change().fillna(0.0)
    pos = weights.shift(1).fillna(0.0)  # SEM look-ahead: peso de t-1 no retorno de t
    turnover = pos.diff().abs().fillna(0.0)
    strat_ret = pos * ret - turnover * fee  # custo proporcional ao giro
    bh_ret = ret  # benchmark = buy & hold (peso 1)

    strat_ret.index = pd.to_datetime(strat_ret.index)
    bh_ret.index = pd.to_datetime(bh_ret.index)

    if to_daily:  # corrige anualização em dados intradiários
        strat_ret = (1 + strat_ret).resample("1D").prod() - 1
        bh_ret = (1 + bh_ret).resample("1D").prod() - 1

    return strat_ret, bh_ret


def _basic_metrics(label: str, ret: pd.Series, periods_per_year: float) -> dict:
    """Métricas simples (sem quantstats): retorno, vol., Sharpe aproximado e
    maxDD. Servem de (a) fallback quando o relatório do quantstats quebra
    (ex.: variância zero — comum num agente pouco treinado que nunca opera) e
    (b) fonte do --metrics-out (Fase 8.8: comparação entre braços do ensemble
    precisa de números, não do texto do relatório completo)."""
    cumulative = float((1 + ret).prod() - 1)
    vol = float(ret.std())
    ann_vol = vol * (periods_per_year**0.5)
    mean_ret = float(ret.mean())
    sharpe = (mean_ret / vol) * (periods_per_year**0.5) if vol > 0 else float("nan")
    equity = (1 + ret).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1).min())
    return {
        "label": label,
        "retorno_acumulado": cumulative,
        "volatilidade_anualizada": ann_vol,
        "sharpe_aproximado": sharpe,
        "max_drawdown": max_drawdown,
    }


def metrics_dataframe(strat: pd.Series, bh: pd.Series, periods_per_year: float) -> pd.DataFrame:
    """DataFrame com uma linha 'estrategia' e uma 'buy_and_hold' — usado pelo
    --metrics-out para comparação programática entre braços (Fase 8.8)."""
    return pd.DataFrame(
        [
            _basic_metrics("estrategia", strat, periods_per_year),
            _basic_metrics("buy_and_hold", bh, periods_per_year),
        ]
    )


def report_metrics(strat: pd.Series, bh: pd.Series, periods_per_year: float) -> None:
    """Tenta o relatório completo do quantstats; se ele quebrar (ex.: séries com
    variância zero, comum num agente ainda pouco treinado que nunca opera),
    cai para métricas básicas em vez de derrubar o pipeline inteiro."""
    try:
        print(qs.reports.metrics(strat, benchmark=bh, mode="full", display=False))
    except Exception as e:  # noqa: BLE001 — degradação intencional, não é bug a propagar
        print(f"aviso: qs.reports.metrics falhou ({e}) — usando métricas básicas de fallback")
        print(metrics_dataframe(strat, bh, periods_per_year))


def save_tearsheet(strat: pd.Series, bh: pd.Series, out: str, title: str) -> None:
    try:
        qs.reports.html(strat, benchmark=bh, output=out, title=title)
        print(f"\nTear sheet salvo em {out}")
    except Exception as e:  # noqa: BLE001 — mesma degradação intencional acima
        print(f"aviso: qs.reports.html falhou ({e}) — tear sheet HTML não gerado")
        print("ver métricas básicas de fallback impressas acima")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--weights", required=True, help="CSV date,weight (saída do signals.py)")
    p.add_argument("--fee", type=float, default=0.001, help="custo por unidade de turnover")
    p.add_argument("--to-daily", action="store_true", help="reamostra p/ diária antes das métricas")
    p.add_argument("--out", default="results/tearsheet.html")
    p.add_argument("--metrics-out", default=None,
                   help="opcional: caminho de um CSV com retorno/vol/Sharpe/maxDD de "
                        "estrategia e buy_and_hold (Fase 8.8: comparação entre braços)")
    args = p.parse_args()

    trade = pd.read_csv(args.trade)
    prices = trade.groupby("date")["close"].first().sort_index()
    weights = pd.read_csv(args.weights).set_index("date")["weight"].sort_index()

    strat, bh = run_backtest(prices, weights, fee=args.fee, to_daily=args.to_daily)

    if args.to_daily:
        periods_per_year = 365.0  # cripto negocia todo dia do ano (sem calendário de bolsa)
    else:
        bar_seconds = np.median(np.diff(strat.index.values)).astype("timedelta64[s]").astype(float)
        periods_per_year = (365 * 24 * 60 * 60) / bar_seconds if bar_seconds > 0 else 365.0

    report_metrics(strat, bh, periods_per_year)
    save_tearsheet(strat, bh, args.out, title="Estratégia DRL vs Buy & Hold (BTC)")

    if args.metrics_out:
        metrics_dataframe(strat, bh, periods_per_year).to_csv(args.metrics_out, index=False)
        print(f"métricas (csv) salvas em {args.metrics_out}")


if __name__ == "__main__":
    main()
