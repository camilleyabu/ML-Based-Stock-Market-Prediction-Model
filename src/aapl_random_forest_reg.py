import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from datetime import datetime, timezone, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

if API_KEY is None or API_SECRET is None:
    raise SystemExit("Missing API keys. Copy .env.example -> .env and fill in keys.")

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

    df = api.get_bars(symbol, TimeFrame.Day, start=start, end=end, feed="iex").df
    df = df.reset_index()

    # ---- Features ----
    df["return_1"] = df["close"].pct_change()
    df["return_2"] = df["close"].pct_change(2)
    df["return_5"] = df["close"].pct_change(5)

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    df["vol_10"] = df["return_1"].rolling(10).std()
    df["rsi_14"] = rsi(df["close"], 14)

    # ---- Target (tomorrow return) ----
    df["target_return_next"] = df["close"].pct_change().shift(-1)

    df = df.dropna().copy()

    features = ["return_1", "return_2", "return_5", "sma_10", "sma_20", "vol_10", "rsi_14"]
    X = df[features]
    y = df["target_return_next"]

    n = len(df)
    start_i = int(n * 0.6)
    step = 50

    accs = []
    for i in range(start_i, n - step, step):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i:i+step], y.iloc[i:i+step]

        model = RandomForestRegressor(
            n_estimators=1000,
            random_state=42,
            min_samples_leaf=5,
            max_depth=8
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        actual_dir = (y_test.values > 0).astype(int)
        pred_dir = (pred > 0).astype(int)
        accs.append(accuracy_score(actual_dir, pred_dir))

    print("Walk-forward direction accuracy (mean):", sum(accs) / len(accs))
    print("Walk-forward direction accuracy (min/max):", min(accs), max(accs))


    # ---- Time split (80/20) ----
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ---- Train ----
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    # ---- Predict ----
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    print("MAE (next-day return):", mae)

    # Direction accuracy
    actual_dir = (y_test.values > 0).astype(int)
    pred_dir = (pred > 0).astype(int)
    direction_acc = (actual_dir == pred_dir).mean()
    print("Direction accuracy:", direction_acc)

    tp = ((pred_dir == 1) & (actual_dir == 1)).sum()
    tn = ((pred_dir == 0) & (actual_dir == 0)).sum()
    fp = ((pred_dir == 1) & (actual_dir == 0)).sum()
    fn = ((pred_dir == 0) & (actual_dir == 1)).sum()
    print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)

    baseline_dir = (y_test.shift(1).values > 0).astype(int)  # "tomorrow = same direction as yesterday"
    actual_dir = (y_test.values > 0).astype(int)

    baseline_acc = (baseline_dir == actual_dir).mean()
    print("Baseline (yesterday direction) accuracy:", baseline_acc)
    print("Model direction accuracy:", direction_acc)

    # ---- Save outputs for GitHub showcase ----
    out = df.iloc[split:][["timestamp", "close"]].copy()
    out["actual_next_return"] = y_test.values
    out["pred_next_return"] = pred
    out["actual_dir"] = actual_dir
    out["pred_dir"] = pred_dir
    out["pred_next_close"] = out["close"] * (1 + out["pred_next_return"])

    os.makedirs("outputs", exist_ok=True)
    out.to_csv("outputs/predictions.csv", index=False)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv("outputs/feature_importances.csv")

    # Plot: actual vs predicted returns
    plt.figure()
    plt.plot(out["timestamp"], out["actual_next_return"], label="Actual")
    plt.plot(out["timestamp"], out["pred_next_return"], label="Predicted")
    plt.legend()
    plt.title(f"{symbol} Next-Day Return: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/returns_plot.png", dpi=200)
    plt.show()

    print("Saved: outputs/predictions.csv, outputs/feature_importances.csv, outputs/returns_plot.png")


if __name__ == "__main__":
    main()
