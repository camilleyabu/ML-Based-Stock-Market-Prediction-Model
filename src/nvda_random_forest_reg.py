import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score


def main():
    # =========================
    # Load API keys
    # =========================
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")

    if API_KEY is None or API_SECRET is None:
        raise SystemExit("Missing API keys. Check .env for API_KEY and API_SECRET.")

    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    # =========================
    # Fetch NVDA + SPY + QQQ
    # =========================
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2026, 2, 3)

    symbols = ["NVDA", "SPY", "QQQ"]
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=START_DATE,
        end=END_DATE,
        feed="iex",
    )

    bars = client.get_stock_bars(request_params).df.reset_index()

    nvda = bars[bars["symbol"] == "NVDA"].sort_values("timestamp").reset_index(drop=True)
    spy = bars[bars["symbol"] == "SPY"].sort_values("timestamp").reset_index(drop=True)
    qqq = bars[bars["symbol"] == "QQQ"].sort_values("timestamp").reset_index(drop=True)

    # =========================
    # Market feature frames
    # =========================
    spy_feat = spy[["timestamp", "close"]].copy()
    qqq_feat = qqq[["timestamp", "close"]].copy()

    spy_feat["spy_return_1"] = spy_feat["close"].pct_change()
    qqq_feat["qqq_return_1"] = qqq_feat["close"].pct_change()

    spy_feat["spy_vol_10"] = spy_feat["spy_return_1"].rolling(10).std()
    qqq_feat["qqq_vol_10"] = qqq_feat["qqq_return_1"].rolling(10).std()

    spy_feat = spy_feat[["timestamp", "spy_return_1", "spy_vol_10"]]
    qqq_feat = qqq_feat[["timestamp", "qqq_return_1", "qqq_vol_10"]]

    df = (
        nvda.merge(spy_feat, on="timestamp", how="inner")
            .merge(qqq_feat, on="timestamp", how="inner")
            .sort_values("timestamp")
            .reset_index(drop=True)
    )

    # =========================
    # NVDA engineered features
    # =========================
    df["nvda_return_1"] = df["close"].pct_change()
    df["nvda_return_2"] = df["close"].pct_change(2)
    df["nvda_return_5"] = df["close"].pct_change(5)

    df["nvda_vol_5"] = df["nvda_return_1"].rolling(5).std()
    df["nvda_vol_10"] = df["nvda_return_1"].rolling(10).std()

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["volume_change"] = df["volume"].pct_change()
    df["volume_sma_10"] = df["volume"].rolling(10).mean()

    # =========================
    # Target: next-day RETURN
    # =========================
    df["target_return_next"] = df["close"].pct_change().shift(-1)

    df = df.dropna().copy()

    features = [
        "nvda_return_1",
        "nvda_return_2",
        "nvda_return_5",
        "nvda_vol_5",
        "nvda_vol_10",
        "ma_5",
        "ma_10",
        "ma_20",
        "volume_change",
        "volume_sma_10",
        "spy_return_1",
        "spy_vol_10",
        "qqq_return_1",
        "qqq_vol_10",
    ]

    X = df[features]
    y = df["target_return_next"]

    # =========================
    # Walk-forward evaluation (AAPL-style)
    # =========================
    n = len(df)
    start_i = int(n * 0.6)
    step = 50

    accs = []
    for i in range(start_i, n - step, step):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test_wf, y_test_wf = X.iloc[i:i + step], y.iloc[i:i + step]

        wf_model = RandomForestRegressor(
            n_estimators=1000,
            random_state=42,
            min_samples_leaf=5,
            max_depth=8,
            n_jobs=-1,
        )
        wf_model.fit(X_train, y_train)
        wf_preds = wf_model.predict(X_test_wf)

        actual_dir = (y_test_wf.to_numpy() > 0).astype(int)
        pred_dir = (wf_preds > 0).astype(int)
        accs.append(accuracy_score(actual_dir, pred_dir))

    print("Walk-forward direction accuracy (mean):", float(np.mean(accs)))
    print("Walk-forward direction accuracy (min/max):", float(np.min(accs)), float(np.max(accs)))

    # =========================
    # Time split (80/20), no shuffle
    # =========================
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # =========================
    # Train + Predict
    # =========================
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=5,
        max_depth=12,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test).flatten()

    # =========================
    # Metrics (AAPL-style)
    # =========================
    mae = mean_absolute_error(y_test, preds)
    print("MAE (next-day return):", float(mae))

    actual_dir = (y_test.to_numpy() > 0).astype(int)
    pred_dir = (preds > 0).astype(int)
    direction_acc = (actual_dir == pred_dir).mean()
    print("Direction accuracy:", float(direction_acc))

    tp = int(((pred_dir == 1) & (actual_dir == 1)).sum())
    tn = int(((pred_dir == 0) & (actual_dir == 0)).sum())
    fp = int(((pred_dir == 1) & (actual_dir == 0)).sum())
    fn = int(((pred_dir == 0) & (actual_dir == 1)).sum())
    print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)

    # Baseline: yesterday direction
    baseline_dir = (y.iloc[split - 1: -1].to_numpy() > 0).astype(int)  # aligns with test length
    baseline_acc = (baseline_dir == actual_dir).mean()
    print("Baseline (yesterday direction) accuracy:", float(baseline_acc))
    print("Model direction accuracy:", float(direction_acc))

    # =========================
    # Save outputs (same as AAPL)
    # =========================
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    out = df.iloc[split:][["timestamp", "close"]].copy()
    out["actual_next_return"] = y_test.to_numpy()
    out["pred_next_return"] = preds
    out["actual_dir"] = actual_dir
    out["pred_dir"] = pred_dir
    out["pred_next_close"] = out["close"] * (1 + out["pred_next_return"])

    out.to_csv(out_dir / "nvda_predictions.csv", index=False)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv(out_dir / "nvda_feature_importances.csv")

    # =========================
    # Plot (match your LSTM plot style)
    # =========================
    x = np.arange(len(y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(x, y_test.to_numpy(), label="Actual")
    plt.plot(x, preds, label="Predicted")

    plt.legend(loc="upper left")
    plt.title("NVDA Next-Day Return Prediction (Random Forest)")
    plt.xlabel("Test Time Index")
    plt.ylabel("Return")
    plt.tight_layout()

    plot_path = out_dir / "nvda_random_forest_predictions_plot.png"
    plt.savefig(plot_path, dpi=200)
    print("Saved plot to:", plot_path)
    plt.show()


if __name__ == "__main__":
    main()