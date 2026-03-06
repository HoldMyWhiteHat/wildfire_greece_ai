import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path


# CONFIG

DATA_PATH = "data/processed/features_daily1.csv"
MODEL_PATH = "src/models/lstm_fire_model2.pt"

SEQ_LEN = 5
BATCH_SIZE = 64
EPOCHS = 15 
LR = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 1

TARGET_COL = "fire_risk"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# DATASET

class FireDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# MODEL

class FireLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_embedding=False):
        out, _ = self.lstm(x)
        embedding = out[:, -1, :]  # (batch, hidden_dim)

        logits = self.fc(embedding)
        probs = torch.sigmoid(logits).squeeze()

        if return_embedding:
            return probs, embedding

        return probs


# SEQUENCE CREATION

def build_sequences(df, feature_cols, max_per_cell=400):
    X, y = [], []

    for (_, _), g in df.groupby(["latitude", "longitude"]):
        g = g.sort_values("date")

        feats = g[feature_cols].values.astype(np.float32)
        labels = g[TARGET_COL].values.astype(np.float32)

        max_i = len(g) - SEQ_LEN
        if max_i <= 0:
            continue

        # sequences per grid cell
        idxs = np.arange(max_i)
        if len(idxs) > max_per_cell:
            idxs = np.random.choice(idxs, max_per_cell, replace=False)

        for i in idxs:
            X.append(feats[i:i + SEQ_LEN])
            y.append(labels[i + SEQ_LEN])

        

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)




# TRAIN

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

  
    # Date handling
    
    df["date"] = pd.to_datetime(df["date"])

    
    # Feature selection
    
    NON_FEATURES = ["date", "latitude", "longitude", TARGET_COL]
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]

    
    # HARD FIX: remove strings
    
    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # Safety check
    bad = df[feature_cols].select_dtypes(exclude=["number"]).columns
    if len(bad) > 0:
        raise ValueError(f"Non-numeric columns found: {bad}")

    
    # Scaling
   
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    
    # Build sequences
   
    X, y = build_sequences(df, feature_cols)

    print("Sequences:", X.shape)
    print("Positive labels:", int(y.sum()))
    MAX_SAMPLES = 500_000  # plenty for LSTM
    if len(X) > MAX_SAMPLES:
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[idx]
        y = y[idx]    

    
    # Train / validation split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_loader = DataLoader(
        FireDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        FireDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    
    # Model
    
    model = FireLSTM(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print("Training started")

    
    # Training loop
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_loss += criterion(preds, yb).item()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )

    
    # Save model
    
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved")

    
    # Sanity check
    
    model.eval()
    with torch.no_grad():
        probs = model(
            torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    print("\nSanity check:")
    print("Mean prob:", probs.mean())
    print("Max prob:", probs.max())
    print("Predictions > 0.01:", (probs > 0.01).sum())
    print("Predictions > 0.05:", (probs > 0.05).sum())
    print("Predictions > 0.5:", (probs > 0.5).sum())


# ENTRY POINT

if __name__ == "__main__":
    train()