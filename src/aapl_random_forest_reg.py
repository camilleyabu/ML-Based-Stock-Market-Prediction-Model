import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score


load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

if API_KEY is None or API_SECRET is None:
    raise SystemExit("Missing API keys. Check .env for API_KEY and API_SECRET.")

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def main():
    symbol = "AAPL"
    start = "2022-01-01"
    end = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    df = api.get_bars(symbol, TimeFrame.Day, start=start, end=end, feed="iex").df.reset_index()

    # =========================
    # Features
    # =========================
    df["return_1"] = df["close"].pct_change()
    df["return_2"] = df["close"].pct_change(2)
    df["return_5"] = df["close"].pct_change(5)

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    df["vol_10"] = df["return_1"].rolling(10).std()
    df["rsi_14"] = rsi(df["close"], 14)

    # Target: next-day return
    df["target_return_next"] = df["close"].pct_change().shift(-1)

    df = df.dropna().copy()

    features = ["return_1", "return_2", "return_5", "sma_10", "sma_20", "vol_10", "rsi_14"]
    X = df[features]
    y = df["target_return_next"]

    # =========================
    # Walk-forward evaluation (optional, prints summary)
    # =========================
    n = len(df)
    start_i = int(n * 0.6)
    step = 50

    accs = []
    for i in range(start_i, n - step, step):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i:i + step], y.iloc[i:i + step]

        wf_model = RandomForestRegressor(
            n_estimators=1000,
            random_state=42,
            min_samples_leaf=5,
            max_depth=8,
            n_jobs=-1,
        )
        wf_model.fit(X_train, y_train)
        wf_preds = wf_model.predict(X_test)

        actual_dir = (y_test.to_numpy() > 0).astype(int)
        pred_dir = (wf_preds > 0).astype(int)
        accs.append(accuracy_score(actual_dir, pred_dir))

    print("Walk-forward direction accuracy (mean):", float(np.mean(accs)))
    print("Walk-forward direction accuracy (min/max):", float(np.min(accs)), float(np.max(accs)))

    # =========================
    # Time split (80/20)
    # =========================
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # =========================
    # Train + Predict
    # =========================
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test).flatten()

    # =========================
    # Metrics
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

    # Baseline: "tomorrow direction = yesterday direction"
    # For the test window, yesterday's direction is the previous day's y (shifted by 1)
    baseline_dir = (y.iloc[split - 1: -1].to_numpy() > 0).astype(int)  # aligns length with test set
    baseline_acc = (baseline_dir == actual_dir).mean()
    print("Baseline (yesterday direction) accuracy:", float(baseline_acc))
    print("Model direction accuracy:", float(direction_acc))

    # =========================
    # Save outputs (consistent project-root outputs/)
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

    out.to_csv(out_dir / "aapl_predictions.csv", index=False)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv(out_dir / "aapl_feature_importances.csv")

    # =========================
    # Plot: match your LSTM plot style exactly
    # =========================
    x = np.arange(len(y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(x, y_test.to_numpy(), label="Actual")
    plt.plot(x, preds, label="Predicted")

    plt.legend(loc="upper left")
    plt.title("AAPL Next-Day Return Prediction (Random Forest)")
    plt.xlabel("Test Time Index")
    plt.ylabel("Return")
    plt.tight_layout()

    plot_path = out_dir / "aapl_random_forest_predictions_plot.png"
    plt.savefig(plot_path, dpi=200)
    print("Saved plot to:", plot_path)

    plt.show()


if __name__ == "__main__":
    main()