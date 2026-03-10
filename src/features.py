"""
Shared feature engineering for all ML stock prediction models.
All models (Random Forest, LSTM, Transformer) use the same feature set
to ensure fair comparison.
"""

import csv
from datetime import datetime, date
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


FEATURE_COLS = [
    # Lagged returns
    "return_1", "return_2", "return_3", "return_5",
    # Moving averages
    "sma_5", "sma_10", "sma_20",
    # Rolling volatility
    "vol_5", "vol_10",
    # RSI momentum
    "rsi_14",
    # MACD trend
    "macd", "macd_signal",
    # Bollinger Band width (normalized volatility)
    "bb_width",
    # Volume activity
    "volume_change", "volume_sma_10",
    # Broad market context
    "spy_return_1", "spy_vol_10",
    # Tech market context
    "qqq_return_1", "qqq_vol_10",
]


def fetch_bars(api_key: str, api_secret: str, symbols: list,
               start: datetime, end: datetime) -> dict:
    """Fetch daily OHLCV bars for symbols via Alpaca API (IEX feed)."""
    client = StockHistoricalDataClient(api_key, api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",
    )
    bars = client.get_stock_bars(request).df.reset_index()
    return {
        sym: bars[bars["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        for sym in symbols
    }


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta).clip(lower=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def build_features(df_stock: pd.DataFrame, df_spy: pd.DataFrame,
                   df_qqq: pd.DataFrame) -> pd.DataFrame:
    """
    Build standardized features for a stock using SPY and QQQ as market context.

    All DataFrames must have columns: timestamp, close, volume.
    Returns a DataFrame with FEATURE_COLS + target_return_next, NaN rows dropped.
    Columns: timestamp, close, <all FEATURE_COLS>, target_return_next
    """
    # Market context features (fit before merge to avoid polluting stock df)
    spy = df_spy[["timestamp", "close"]].copy()
    spy["spy_return_1"] = spy["close"].pct_change()
    spy["spy_vol_10"] = spy["spy_return_1"].rolling(10).std()
    spy = spy[["timestamp", "spy_return_1", "spy_vol_10"]]

    qqq = df_qqq[["timestamp", "close"]].copy()
    qqq["qqq_return_1"] = qqq["close"].pct_change()
    qqq["qqq_vol_10"] = qqq["qqq_return_1"].rolling(10).std()
    qqq = qqq[["timestamp", "qqq_return_1", "qqq_vol_10"]]

    df = (
        df_stock
        .merge(spy, on="timestamp", how="inner")
        .merge(qqq, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    close = df["close"]

    # Lagged returns
    df["return_1"] = close.pct_change(1)
    df["return_2"] = close.pct_change(2)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    # Moving averages
    df["sma_5"]  = close.rolling(5).mean()
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()

    # Rolling volatility
    df["vol_5"]  = df["return_1"].rolling(5).std()
    df["vol_10"] = df["return_1"].rolling(10).std()

    # RSI
    df["rsi_14"] = _rsi(close, 14)

    # MACD
    df["macd"], df["macd_signal"] = _macd(close)

    # Bollinger Band width: (upper - lower) / mid = 4*std / sma_20 (normalized)
    bb_std = close.rolling(20).std()
    df["bb_width"] = (4 * bb_std) / df["sma_20"]

    # Volume
    df["volume_change"]  = df["volume"].pct_change()
    df["volume_sma_10"]  = df["volume"].rolling(10).mean()

    # Target: next-day return
    df["target_return_next"] = close.pct_change().shift(-1)

    keep = ["timestamp", "close"] + FEATURE_COLS + ["target_return_next"]
    return df[keep].dropna().reset_index(drop=True)


def save_metrics(out_dir: Path, ticker: str, model_name: str,
                 mae: float, direction_acc: float,
                 baseline_acc: float, n_test: int) -> None:
    """
    Upsert a row in outputs/results_summary.csv for ticker + model.
    Overwrites the existing row if one exists, so re-running never duplicates.
    """
    metrics_file = out_dir / "results_summary.csv"
    fieldnames = ["ticker", "model", "mae", "direction_accuracy",
                  "baseline_accuracy", "n_test", "run_date"]

    new_row = {
        "ticker": ticker,
        "model": model_name,
        "mae": round(mae, 5),
        "direction_accuracy": round(direction_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4) if baseline_acc is not None else "",
        "n_test": n_test,
        "run_date": date.today().isoformat(),
    }

    rows = []
    if metrics_file.exists():
        with open(metrics_file, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r["ticker"] == ticker and r["model"] == model_name)]

    rows.append(new_row)
    rows.sort(key=lambda r: (r["ticker"], r["model"]))

    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metrics saved → {metrics_file}")
