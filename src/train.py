import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from src.model import StockMovementModel


def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(config):
    proc_dir = os.path.dirname(config.get('processed_data_path', 'data/processed/processed_data.csv'))
    # Paths for CSV splits
    paths = {
        'X_train': os.path.join(proc_dir, 'X_train.csv'),
        'y_train': os.path.join(proc_dir, 'y_train.csv'),
        'X_val':   os.path.join(proc_dir, 'X_val.csv'),
        'y_val':   os.path.join(proc_dir, 'y_val.csv')
    }
    # Load as numpy
    data = {}
    for key, path in paths.items():
        df = pd.read_csv(path)
        data[key] = df.values
    return data


def create_data_loaders(data, batch_size):
    # Convert to torch tensors
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'].reshape(-1), dtype=torch.float32)
    X_val   = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val   = torch.tensor(data['y_val'].reshape(-1), dtype=torch.float32)

    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train():
    # Load config and hyperparams
    config = load_config()
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 32)
    epochs     = train_cfg.get('epochs', 20)
    lr         = train_cfg.get('learning_rate', 1e-3)
    save_path  = train_cfg.get('save_path', 'models/best_model.pth')
    patience   = train_cfg.get('early_stopping_patience', 5)

    # Prepare data
    splits = load_splits(config)
    train_loader, val_loader = create_data_loaders(splits, batch_size)

    # Model
    input_dim = splits['X_train'].shape[1]
    model = StockMovementModel(input_dim=input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze()
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())
                all_preds.extend((preds.cpu() > 0.5).int().tolist())
                all_labels.extend(y_batch.cpu().int().tolist())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss   = sum(val_losses) / len(val_losses)
        val_acc        = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch}/{epochs}  Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Training complete. Best model saved to {save_path}")


if __name__ == '__main__':
    train()

