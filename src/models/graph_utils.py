import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

DATA_PATH = "data/processed/features_daily1.csv"

# STEP 1 — Build Nodes


def build_nodes(df):
    """
    Create graph nodes from unique geographic locations.

    Returns:
    - node_df: DataFrame with [latitude, longitude, node_id]
    - node_id_map: dict {(lat, lon): node_id}
    """

    node_df = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude"])
        .reset_index(drop=True)
    )

    node_df["node_id"] = node_df.index.astype(int)

    node_id_map = {
        (row.latitude, row.longitude): row.node_id
        for _, row in node_df.iterrows()
    }

    return node_df, node_id_map



# STEP 2 — Build Edges (kNN spatial graph)


def build_edges(node_df, k=6):
    """
    Build spatial edges using k-nearest neighbors.

    Returns:
    - edge_index: torch.LongTensor [2, num_edges]
    """

    coords = node_df[["latitude", "longitude"]].values

    # Convert to radians for haversine
    coords_rad = np.radians(coords)

    knn = NearestNeighbors(
        n_neighbors=k + 1,  # +1 for self
        metric="haversine"
    )
    knn.fit(coords_rad)

    _, indices = knn.kneighbors(coords_rad)

    edge_list = []

    for src, neighbors in enumerate(indices):
        for dst in neighbors[1:]:  # skip self-loop
            edge_list.append((src, dst))
            edge_list.append((dst, src))  # undirected graph

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index



# STEP 3 — Run Step 2


def run_step_2(df, k=6):
    node_df, node_id_map = build_nodes(df)
    edge_index = build_edges(node_df, k=k)

    print("Graph built successfully")
    print(f"Nodes: {len(node_df)}")
    print(f"Edges: {edge_index.shape[1]}")

    # Sanity checks
    assert edge_index.shape[0] == 2
    assert edge_index.min() >= 0
    assert edge_index.max() < len(node_df)

    return node_df, node_id_map, edge_index



# EXAMPLE USAGE

if __name__ == "__main__":
    # Example: load your daily features file
    df = pd.read_csv(DATA_PATH)

    node_df, node_id_map, edge_index = run_step_2(df, k=6)

    # Optional: save artifacts
    node_df.to_csv("src/models/graph_nodes.csv", index=False)
    torch.save(edge_index, "src/models/graph_edge_index.pt")
