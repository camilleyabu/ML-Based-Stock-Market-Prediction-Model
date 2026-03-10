import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

from features import fetch_bars, build_features, FEATURE_COLS, save_metrics


def main():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")
    if not API_KEY or not API_SECRET:
        raise SystemExit("Missing API keys. Check .env for API_KEY and API_SECRET.")

    symbol = "NVDA"
    bars = fetch_bars(API_KEY, API_SECRET, [symbol, "SPY", "QQQ"],
                      start=datetime(2022, 1, 1), end=datetime.now())

    df = build_features(bars[symbol], bars["SPY"], bars["QQQ"])

    X = df[FEATURE_COLS]
    y = df["target_return_next"]
    n = len(df)

    # =========================
    # Walk-forward validation
    # =========================
    start_i = int(n * 0.6)
    step = 50
    accs = []
    for i in range(start_i, n - step, step):
        wf_model = RandomForestRegressor(
            n_estimators=1000, random_state=42, min_samples_leaf=5,
            max_depth=10, n_jobs=-1,
        )
        wf_model.fit(X.iloc[:i], y.iloc[:i])
        wf_preds = wf_model.predict(X.iloc[i:i + step])
        actual_dir = (y.iloc[i:i + step].to_numpy() > 0).astype(int)
        pred_dir = (wf_preds > 0).astype(int)
        accs.append(accuracy_score(actual_dir, pred_dir))

    print("Walk-forward direction accuracy (mean):", float(np.mean(accs)))
    print("Walk-forward direction accuracy (min/max):", float(np.min(accs)), float(np.max(accs)))

    # =========================
    # 80/20 chronological split
    # =========================
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=500, random_state=42, min_samples_leaf=5,
        max_depth=10, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test).flatten()

    # =========================
    # Metrics
    # =========================
    mae = mean_absolute_error(y_test, preds)
    actual_dir = (y_test.to_numpy() > 0).astype(int)
    pred_dir = (preds > 0).astype(int)
    direction_acc = (actual_dir == pred_dir).mean()
    baseline_dir = (y.iloc[split - 1:-1].to_numpy() > 0).astype(int)
    baseline_acc = (baseline_dir == actual_dir).mean()

    print(f"MAE (next-day return): {float(mae):.5f}")
    print(f"Direction accuracy:    {float(direction_acc):.4f}")
    print(f"Baseline accuracy:     {float(baseline_acc):.4f}")

    tp = int(((pred_dir == 1) & (actual_dir == 1)).sum())
    tn = int(((pred_dir == 0) & (actual_dir == 0)).sum())
    fp = int(((pred_dir == 1) & (actual_dir == 0)).sum())
    fn = int(((pred_dir == 0) & (actual_dir == 1)).sum())
    print(f"TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")

    # =========================
    # Save outputs
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

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    importances.to_csv(out_dir / "nvda_feature_importances.csv")

    save_metrics(out_dir, "NVDA", "Random Forest", mae, direction_acc, baseline_acc, len(y_test))

    # =========================
    # Plot
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
