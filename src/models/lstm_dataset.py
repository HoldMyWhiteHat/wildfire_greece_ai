import numpy as np
import pandas as pd

WINDOW = 14
FEATURE_COLS = [
    "t2m",
    "wind_speed",
    "d2m",
    "tp",
    "dryness_index",
    "fuel_risk"
]
TARGET_COL = "fire_risk"

def build_lstm_tensors(csv_path, neg_keep_prob=0.1):
    df = pd.read_csv(csv_path, parse_dates=["date"])

    X, y = [], []

    for _, g in df.groupby(["latitude", "longitude"]):
        g = g.sort_values("date")
        values = g[FEATURE_COLS].values
        labels = g[TARGET_COL].values

        for i in range(len(g) - WINDOW):
            label = labels[i + WINDOW - 1]

            # keep all positives, sample negatives
            if label == 0 and np.random.rand() > neg_keep_prob:
                continue

            X.append(values[i:i+WINDOW])
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return X, y