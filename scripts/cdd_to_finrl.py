from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from finrl import config
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split


def _resolve_csv_paths(csv_arg: str) -> list[str]:
    """Aceita um arquivo, uma pasta (lê todo *.csv dentro) ou um glob; devolve caminhos ordenados."""
    if os.path.isdir(csv_arg):
        paths = sorted(glob.glob(os.path.join(csv_arg, "*.csv")))
    elif any(ch in csv_arg for ch in "*?["):
        paths = sorted(glob.glob(csv_arg))
    else:
        paths = [csv_arg]
    if not paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado para: {csv_arg}")
    return paths


def _read_one(csv_path: str) -> pd.DataFrame:
    """Lê um CSV bruto da CDD (1ª linha = comentário) e normaliza para dt/OHLCV."""
    raw = pd.read_csv(csv_path, skiprows=1)
    raw.columns = [c.strip().lower() for c in raw.columns]

    vol_col = next(
        (c for c in raw.columns if c.startswith("volume") and ("btc" in c or "base" in c)),
        None,
    ) or next(c for c in raw.columns if c.startswith("volume"))
    raw["volume"] = raw[vol_col]

    # format="mixed": alguns arquivos da CDD têm timestamps corrompidos misturados com
    # os válidos (linhas com data em 1970, artefato de bug no export da fonte) — parsear
    # elemento a elemento evita que o formato inferido de uma linha corrompida invalide
    # todas as linhas boas do arquivo.
    raw["dt"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    bad = raw["dt"].isna() | (raw["dt"].dt.year < 2015)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"aviso: {n_bad} linha(s) com timestamp inválido/corrompido em {csv_path} descartada(s)")
    raw = raw.loc[~bad]
    return raw[["dt", "open", "high", "low", "close", "volume"]]


def load_cdd(csv_path: str, tic: str = "BTCUSDT", resample: str | None = None) -> pd.DataFrame:
    """Lê um ou mais CSVs (1 minuto) da CryptoDataDownload e devolve o DataFrame no
    contrato do FinRL: date, open, high, low, close, volume, tic.

    csv_path: um arquivo, uma pasta (concatena todo *.csv dentro) ou um glob
    (ex.: 'data/raw/Binance_BTCUSDT_*_minute.csv'). Múltiplos arquivos são
    concatenados, deduplicados por timestamp e ordenados em ordem crescente.
    resample: None mantém 1 minuto. Ou passe uma regra pandas: '5min', '15min', '1h'.
    """
    paths = _resolve_csv_paths(csv_path)
    parts = [_read_one(p) for p in paths]
    raw = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    raw = raw.sort_values("dt").drop_duplicates("dt").set_index("dt")

    if resample:
        raw = raw.resample(resample).agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])

    df = raw.reset_index()
    df["tic"] = tic
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[["date", "open", "high", "low", "close", "volume", "tic"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="arquivo CSV, pasta (ex.: data/raw) ou glob (ex.: 'data/raw/*.csv') da CryptoDataDownload")
    p.add_argument("--tic", default="BTCUSDT")
    p.add_argument("--resample", default=None,
                   help="regra pandas p/ reamostrar: 5min, 15min, 1h. Vazio = 1 minuto puro")
    p.add_argument("--train-out", default="train_data.csv")
    p.add_argument("--val-out", default="val_data.csv")
    p.add_argument("--trade-out", default="trade_data.csv")
    args = p.parse_args()

    df = load_cdd(args.csv, tic=args.tic, resample=args.resample)
    print(f"linhas após leitura/concatenação/reamostragem: {len(df):,}")
    print(f"período coberto: {df.date.min()} -> {df.date.max()}")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # TRADE_END_DATE=None => usa a última data disponível (data_split é exclusivo
    # no limite superior, então soma 1s à data máxima para incluir a última barra).
    if config.TRADE_END_DATE is not None:
        trade_end = config.TRADE_END_DATE
    else:
        max_dt = pd.to_datetime(processed["date"]).max() + pd.Timedelta(seconds=1)
        trade_end = max_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"TRADE_END_DATE não hard-coded — detectado dinamicamente: {trade_end}")

    train = data_split(processed, config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    val = data_split(processed, config.VALIDATION_START_DATE, config.VALIDATION_END_DATE)
    trade = data_split(processed, config.TRADE_START_DATE, trade_end)

    train.to_csv(args.train_out, index=False)
    val.to_csv(args.val_out, index=False)
    trade.to_csv(args.trade_out, index=False)
    print(f"train: {train.shape}  val: {val.shape}  trade: {trade.shape}")
    print(f"período train: {train.date.min()} -> {train.date.max()}")
    print(f"período val:   {val.date.min()} -> {val.date.max()}")
    print(f"período trade: {trade.date.min()} -> {trade.date.max()}")


if __name__ == "__main__":
    main()
