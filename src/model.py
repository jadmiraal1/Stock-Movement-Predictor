
import yaml
import torch
import torch.nn as nn


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class StockMovementModel(nn.Module):
    
    def __init__(self, input_dim: int, config_path: str = "config/config.yaml"):
        super(StockMovementModel, self).__init__()
        # Load model config
        cfg = load_config(config_path).get('model', {})
        hidden_dims = cfg.get('hidden_dims', [128, 64])
        dropout_p = cfg.get('dropout', 0.5)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = h
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Expects input shape (batch_size, input_dim).
        Returns: output shape (batch_size, 1), probability of stock going up.
        """
        return self.network(x)


if __name__ == '__main__':
    # Quick test of model instantiation
    dummy_input_dim = 4  # example feature size
    model = StockMovementModel(input_dim=dummy_input_dim)
    sample = torch.randn(2, dummy_input_dim)
    out = model(sample)
    print(f"Output shape: {out.shape}")
