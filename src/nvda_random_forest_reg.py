import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from pathlib import Path


load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if API_KEY is None or API_SECRET is None:
    raise SystemExit("Missing API keys. Check .env (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 2, 3)

symbols = ["NVDA", "SPY", "QQQ"]

request_params = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
    feed="iex",  # avoids SIP subscription issues
)

bars = client.get_stock_bars(request_params).df.reset_index()

# Split by symbol into separate frames
nvda = bars[bars["symbol"] == "NVDA"].sort_values("timestamp").reset_index(drop=True)
spy  = bars[bars["symbol"] == "SPY"].sort_values("timestamp").reset_index(drop=True)
qqq  = bars[bars["symbol"] == "QQQ"].sort_values("timestamp").reset_index(drop=True)

# Use timestamp as key to merge market features into NVDA
spy_feat = spy[["timestamp", "close"]].copy()
qqq_feat = qqq[["timestamp", "close"]].copy()

spy_feat["spy_return_1"] = spy_feat["close"].pct_change()
qqq_feat["qqq_return_1"] = qqq_feat["close"].pct_change()

spy_feat["spy_vol_10"] = spy_feat["spy_return_1"].rolling(10).std()
qqq_feat["qqq_vol_10"] = qqq_feat["qqq_return_1"].rolling(10).std()

spy_feat = spy_feat[["timestamp", "spy_return_1", "spy_vol_10"]]
qqq_feat = qqq_feat[["timestamp", "qqq_return_1", "qqq_vol_10"]]

df = nvda.merge(spy_feat, on="timestamp", how="inner").merge(qqq_feat, on="timestamp", how="inner")

# ---- NVDA features ----
df["nvda_return_1"] = df["close"].pct_change()
df["nvda_vol_5"] = df["nvda_return_1"].rolling(5).std()

df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

df["volume_change"] = df["volume"].pct_change()
df["volume_sma_10"] = df["volume"].rolling(10).mean()

# Target: NVDA next-day close
df["target_close_next"] = df["close"].shift(-1)

df = df.dropna().copy()

features = [
    "close",
    "volume",
    "volume_change",
    "volume_sma_10",
    "nvda_vol_5",
    "ma_5",
    "ma_10",
    "ma_20",
    "spy_return_1",
    "spy_vol_10",
    "qqq_return_1",
    "qqq_vol_10",
]

X = df[features]
y = df["target_close_next"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: ${mae:.2f}")

# Direction accuracy (next day close vs today's close)
today_close_test = df.loc[X_test.index, "close"].values
actual_dir = (y_test.values > today_close_test).astype(int)
pred_dir = (preds > today_close_test).astype(int)
print("Direction accuracy:", (actual_dir == pred_dir).mean())

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Next Close")
plt.plot(preds, label="Predicted Next Close")
plt.title("NVDA Next-Day Close Prediction (with SPY + QQQ features)")
plt.xlabel("Test Time Index")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()

project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "outputs"
out_dir.mkdir(exist_ok=True)

plot_path = out_dir / "nvda_predictions_plot.png"
plt.savefig(plot_path, dpi=200)
print("Saved plot to:", plot_path)

plt.show()

# Predict next day close for the most recent row
latest_X = X.iloc[-1:].values
next_close_pred = model.predict(latest_X)[0]
print(f"Next Day Predicted Close for NVDA: ${next_close_pred:.2f}")
