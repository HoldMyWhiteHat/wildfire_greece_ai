import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd


# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NODE_FEATURE_PATH = "src/models/node_features.pt"
EDGE_INDEX_PATH = "src/models/graph_edge_index.pt"
NODE_DF_PATH = "src/models/graph_nodes.csv"
DATA_PATH = "data/processed/features_daily1.csv"

EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 64



# GNN MODEL

class FireGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return torch.sigmoid(self.out(x)).squeeze()



# LOAD DATA

print("Loading graph data...")

x = torch.load(NODE_FEATURE_PATH).to(DEVICE)
edge_index = torch.load(EDGE_INDEX_PATH).to(DEVICE)
node_df = pd.read_csv(NODE_DF_PATH)

df = pd.read_csv(DATA_PATH)


# BUILD NODE LABELS
# Label = mean fire risk per node (spatial supervision)
labels = (
    df.groupby(["latitude", "longitude"])["fire_risk"]
    .mean()
    .reset_index()
)

node_labels = []
for _, row in node_df.iterrows():
    match = labels[
        (labels.latitude == row.latitude) &
        (labels.longitude == row.longitude)
    ]
    if len(match) == 0:
        node_labels.append(0.0)
    else:
        node_labels.append(match.fire_risk.values[0])

y = torch.tensor(node_labels, dtype=torch.float32).to(DEVICE)


# GRAPH OBJECT

data = Data(x=x, edge_index=edge_index, y=y).to(DEVICE)


# TRAIN

def train():
    model = FireGNN(
        input_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print("Training GNN...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(data.x, data.edge_index)
        loss = criterion(preds, data.y)

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

   
    # SAVE MODEL
    
    torch.save(model.state_dict(), "src/models/gnn_fire_model.pt")
    print("GNN model saved")

if __name__ == "__main__":
    train()