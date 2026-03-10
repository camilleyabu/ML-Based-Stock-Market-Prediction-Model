import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.callbacks import EarlyStopping

from features import fetch_bars, build_features, FEATURE_COLS, save_metrics
from transformer_model import build_transformer


load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing API keys. Check .env for API_KEY and API_SECRET.")

# =========================
# Fetch and build features
# =========================
bars = fetch_bars(API_KEY, API_SECRET, ["AAPL", "SPY", "QQQ"],
                  start=datetime(2022, 1, 1), end=datetime.now())
df = build_features(bars["AAPL"], bars["SPY"], bars["QQQ"])

X_all = df[FEATURE_COLS].values
y_all = df["target_return_next"].values
n = len(df)

# =========================
# 80/20 chronological split
# =========================
split_idx = int(n * 0.8)

# Fit scaler only on training data to prevent leakage
scaler = StandardScaler()
scaler.fit(X_all[:split_idx])
X_scaled = scaler.transform(X_all)

# =========================
# Create sliding-window sequences
# Sequence i → features rows [i : i+SEQ_LEN], label = y[i+SEQ_LEN]
# Test sequences label df rows [split_idx : n]
# =========================
SEQ_LEN = 20

X_seq, y_seq = [], []
for i in range(n - SEQ_LEN):
    X_seq.append(X_scaled[i:i + SEQ_LEN])
    y_seq.append(y_all[i + SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

seq_split = split_idx - SEQ_LEN
X_train, X_test = X_seq[:seq_split], X_seq[seq_split:]
y_train, y_test = y_seq[:seq_split], y_seq[seq_split:]

# =========================
# Build and train Transformer
# =========================
n_features = len(FEATURE_COLS)
model = build_transformer(seq_len=SEQ_LEN, n_features=n_features)
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1,
)

# =========================
# Predict + metrics
# =========================
pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, pred)
actual_dir = (y_test > 0).astype(int)
pred_dir = (pred > 0).astype(int)
direction_acc = float(np.mean(actual_dir == pred_dir))

# Baseline: "tomorrow direction = yesterday direction"
baseline_dir = (y_all[split_idx - 1:-1] > 0).astype(int)
baseline_acc = float(np.mean(baseline_dir == actual_dir))

print(f"Transformer MAE:                {mae:.5f}")
print(f"Transformer Direction Accuracy: {direction_acc:.4f}")
print(f"Baseline accuracy:              {baseline_acc:.4f}")

# =========================
# Save outputs
# =========================
project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "outputs"
out_dir.mkdir(exist_ok=True)

test_rows = df.iloc[split_idx:split_idx + len(y_test)].reset_index(drop=True)
out = test_rows[["timestamp", "close"]].copy()
out["actual_next_return"] = y_test
out["pred_next_return"] = pred
out["actual_dir"] = actual_dir
out["pred_dir"] = pred_dir
out["pred_next_close"] = out["close"] * (1 + out["pred_next_return"])
out.to_csv(out_dir / "aapl_transformer_predictions.csv", index=False)

save_metrics(out_dir, "AAPL", "Transformer", mae, direction_acc, baseline_acc, len(y_test))

# =========================
# Plot
# =========================
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("AAPL Next-Day Return Prediction (Transformer)")
plt.xlabel("Test Time Index")
plt.ylabel("Return")
plt.tight_layout()
plot_path = out_dir / "aapl_transformer_predictions_plot.png"
plt.savefig(plot_path, dpi=200)
print("Saved plot to:", plot_path)
plt.show()
