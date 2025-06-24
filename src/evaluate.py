import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

from src.model import StockMovementModel


def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(config):
    """
    Loads the test feature and label CSVs produced by FeatureEngineer.
    Returns numpy arrays X_test, y_test.
    """
    proc_path = config.get('processed_data_path', 'data/processed/processed_data.csv')
    data_dir = os.path.dirname(proc_path)
    X_test_path = os.path.join(data_dir, 'X_test.csv')
    y_test_path = os.path.join(data_dir, 'y_test.csv')

    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError('Test data files not found in data/processed')

    X_test = pd.read_csv(X_test_path).values
    y_test = pd.read_csv(y_test_path)['label'].values
    return X_test, y_test


def evaluate():
    # Load config
    config = load_config()
    model_cfg = config.get('training', {})
    save_path = model_cfg.get('save_path', 'models/best_model.pth')

    # Load test data
    X_test, y_test = load_test_data(config)

    # Setup model
    input_dim = X_test.shape[1]
    model = StockMovementModel(input_dim=input_dim)
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model weights not found at {save_path}")
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.eval()

    # Predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = model(X_tensor).squeeze().numpy()
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, probs)
    except ValueError:
        roc_auc = float('nan')

    print("Evaluation Results:")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC AUC:     {roc_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['Down (0)', 'Up (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label', title='Confusion Matrix')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Save plot
    plots_dir = os.path.join(data_dir if (data_dir := os.path.dirname(config.get('processed_data_path','data/processed/processed_data.csv'))) else '.', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    fig.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # ROC Curve
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")
        roc_path = os.path.join(plots_dir, 'roc_curve.png')
        fig2.savefig(roc_path)
        print(f"ROC curve saved to {roc_path}")
    except Exception:
        print("Unable to plot ROC curve.")


if __name__ == '__main__':
    evaluate()

