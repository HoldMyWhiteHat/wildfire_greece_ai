import torch
import pandas as pd
import numpy as np

from train_lstm import FireLSTM
from train_gnn import FireGNN
#from torch_geometric.data import Data


def build_last_sequence_per_location(df, feature_cols, seq_len):
    sequences = {}

    for (lat, lon), g in df.groupby(["latitude", "longitude"]):
        g = g.sort_values("date")

        if len(g) < seq_len:
            continue

        values = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values
            .astype("float32")
        )

        last_seq = values[-seq_len:]

        sequences[(lat, lon)] = torch.tensor(
            last_seq, dtype=torch.float32
        ).unsqueeze(0)  # [1, seq_len, features]

    return sequences

def extract_next_day_node_embeddings(
    df,
    node_df,
    model,
    feature_cols,
    seq_len,
    device
):
    model.eval()
    node_embeddings = []

    seqs_by_loc = build_last_sequence_per_location(
        df, feature_cols, seq_len
    )

    for _, row in node_df.iterrows():
        key = (row.latitude, row.longitude)

        if key not in seqs_by_loc:
            emb = torch.zeros(model.hidden_dim)
        else:
            X = seqs_by_loc[key].to(device)

            with torch.no_grad():
                _, emb = model(X, return_embedding=True)
                emb = emb.squeeze(0)

        node_embeddings.append(emb.cpu())

    return torch.stack(node_embeddings)