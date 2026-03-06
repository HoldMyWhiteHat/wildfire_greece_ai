import torch
import pandas as pd
import numpy as np

from train_lstm import FireLSTM   # import model class


# CONFIG


SEQ_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# Build sequences per location


def build_sequences_per_location(df, feature_cols, seq_len):
    """
    Returns:
        dict[(lat, lon)] -> Tensor [num_sequences, seq_len, num_features]
    """

    sequences = {}

    for (lat, lon), g in df.groupby(["latitude", "longitude"]):
        g = g.sort_values("date")

        values = g[feature_cols].values.astype(np.float32)
        if len(values) < seq_len:
            continue

        seqs = []
        for i in range(len(values) - seq_len):
            seqs.append(values[i : i + seq_len])

        if seqs:
            sequences[(lat, lon)] = torch.tensor(
                np.array(seqs), dtype=torch.float32
            )

    return sequences



# Extract node embeddings


def extract_node_embeddings(
    df,
    node_df,
    model,
    feature_cols,
    seq_len
):
    model.eval()
    node_embeddings = []

    seqs_by_loc = build_sequences_per_location(
        df, feature_cols, seq_len
    )

    for _, row in node_df.iterrows():
        key = (row.latitude, row.longitude)

        if key not in seqs_by_loc:
            # fallback: zero vector
            emb = torch.zeros(model.hidden_dim)
        else:
            X = seqs_by_loc[key].to(DEVICE)

            with torch.no_grad():
                _, emb_seq = model(X, return_embedding=True)
                emb = emb_seq.mean(dim=0)   # aggregate time

        node_embeddings.append(emb.cpu())

    return torch.stack(node_embeddings)



# MAIN


if __name__ == "__main__":
    print("Loading data...")

    df = pd.read_csv("data/processed/features_daily1.csv", parse_dates=["date"])
    NON_FEATURES = ["date", "latitude", "longitude", "fire_risk"]
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    df[feature_cols] = (
    df[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    )
    
    node_df = pd.read_csv("src/models/graph_nodes.csv")

    print("Loading trained LSTM...")
    model = FireLSTM(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=1
    ).to(DEVICE)

    model.load_state_dict(torch.load("src/models/lstm_fire_model2.pt", map_location=DEVICE))

    print("Extracting node embeddings...")
    node_x = extract_node_embeddings(
        df,
        node_df,
        model,
        feature_cols,
        SEQ_LEN
    )

    print("Node features shape:", node_x.shape)

    torch.save(node_x, "src/models/node_features.pt")
    print("Saved node_features.pt")