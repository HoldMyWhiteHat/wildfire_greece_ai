import torch
import pandas as pd
from pathlib import Path

from train_lstm import FireLSTM
from train_gnn import FireGNN
from torch_geometric.data import Data
from temporal_utils import extract_next_day_node_embeddings


# CONFIG

SEQ_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "data/processed/features_daily1.csv"
NODE_PATH = "src/models/graph_nodes.csv"
EDGE_PATH = "src/models/graph_edge_index.pt"

LSTM_PATH = "src/models/lstm_fire_model2.pt"
GNN_PATH = "src/models/gnn_fire_model.pt"

OUT_DIR = Path("src/visualization/predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "fire_risk"
NON_FEATURES = ["date", "latitude", "longitude", TARGET_COL]


# Load data

df_all = pd.read_csv(DATA_PATH, parse_dates=["date"])
node_df = pd.read_csv(NODE_PATH)
edge_index = torch.load(EDGE_PATH)

feature_cols = [c for c in df_all.columns if c not in NON_FEATURES]

df_all[feature_cols] = (
    df_all[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)


# Load models

lstm = FireLSTM(
    input_dim=len(feature_cols),
    hidden_dim=64
).to(DEVICE)
lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
lstm.eval()

gnn = FireGNN(
    input_dim=64,
    hidden_dim=64
).to(DEVICE)
gnn.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))
gnn.eval()


# Loop over dates

dates = sorted(df_all["date"].unique())

for i in range(SEQ_LEN, len(dates) - 1):
    current_date = dates[i]
    predict_date = dates[i + 1]

    print(f"Predicting for {predict_date.date()}")

    df_cut = df_all[df_all["date"] <= current_date]

    node_x = extract_next_day_node_embeddings(
        df_cut,
        node_df,
        lstm,
        feature_cols,
        SEQ_LEN,
        DEVICE
    )

    data = Data(
        x=node_x.to(DEVICE),
        edge_index=edge_index.to(DEVICE)
    )

    with torch.no_grad():
        risk = gnn(data.x, data.edge_index).cpu().numpy()

    out = node_df.copy()
    out["prediction_date"] = predict_date
    out["fire_risk"] = risk

    out.to_csv(
        OUT_DIR / f"fire_risk_{predict_date.date()}.csv",
        index=False
    )

print("All daily predictions generated")