import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from datetime import datetime, timezone, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt

from pathlib import Path


# =========================
# Load data (same as RF)
# =========================

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

api = tradeapi.REST(API_KEY, API_SECRET, "https://paper-api.alpaca.markets", api_version="v2")

symbol = "NVDA"
start = "2022-01-01"
end = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

df = api.get_bars(symbol, TimeFrame.Day, start=start, end=end, feed="iex").df
df = df.reset_index()

# target: next-day return
df["return"] = df["close"].pct_change()
df["target"] = df["return"].shift(-1)

df = df.dropna()

# =========================
# Scale data
# =========================

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(df[["return"]])

# =========================
# Create sequences
# =========================

SEQ_LEN = 20

X = []
y = []

for i in range(len(returns_scaled) - SEQ_LEN):

    X.append(returns_scaled[i:i+SEQ_LEN])
    y.append(df["target"].iloc[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

# =========================
# Train test split
# =========================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# Build LSTM model
# =========================

model = Sequential()

model.add(LSTM(50, input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

# =========================
# Train
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# Predict
# =========================

pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, pred)

print("LSTM MAE:", mae)

# Direction accuracy

actual_dir = (y_test > 0)
pred_dir = (pred > 0)

accuracy = np.mean(actual_dir == pred_dir)

print("LSTM Direction Accuracy:", accuracy)

# =========================
# Plot and SAVE output
# =========================

plt.figure(figsize=(12, 6))

plt.plot(y_test, label="Actual")
plt.plot(pred, label="Predicted")

plt.legend()
plt.title("NVDA Next-Day Return Prediction (LSTM)")
plt.xlabel("Test Time Index")
plt.ylabel("Return")

plt.tight_layout()

# Create outputs folder same way as Random Forest script
project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "outputs"
out_dir.mkdir(exist_ok=True)

# Save plot
plot_path = out_dir / "nvda_lstm_predictions_plot.png"
plt.savefig(plot_path, dpi=200)

print("Saved plot to:", plot_path)

plt.show()