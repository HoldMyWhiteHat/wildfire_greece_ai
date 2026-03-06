import torch
import pandas as pd
import numpy as np

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

TARGET_COL = "fire_risk"
NON_FEATURES = ["date", "latitude", "longitude", TARGET_COL]


# Load data

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
node_df = pd.read_csv(NODE_PATH)
edge_index = torch.load(EDGE_PATH)

feature_cols = [c for c in df.columns if c not in NON_FEATURES]

df[feature_cols] = (
    df[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)


# Load models

lstm = FireLSTM(
    input_dim=len(feature_cols),
    hidden_dim=64
).to(DEVICE)

lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))

gnn = FireGNN(
    input_dim=64,
    hidden_dim=64
).to(DEVICE)

gnn.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))


# Build node embeddings (T → T+1)

node_x = extract_next_day_node_embeddings(
    df,
    node_df,
    lstm,
    feature_cols,
    SEQ_LEN,
    DEVICE
)


# GNN inference

data = Data(
    x=node_x.to(DEVICE),
    edge_index=edge_index.to(DEVICE)
)

gnn.eval()
with torch.no_grad():
    next_day_risk = gnn(data.x, data.edge_index).cpu().numpy()


# Save predictions

out = node_df.copy()
out["fire_risk_tomorrow"] = next_day_risk

out.to_csv("src/models/predictions_next_day.csv", index=False)
print("Next-day fire risk saved to predictions_next_day.csv")