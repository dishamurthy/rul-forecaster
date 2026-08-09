"""
RUL Prediction Models — PyTorch
Architectures:
  1. LSTMForecaster     — Stacked LSTM with MC Dropout
  2. CNN1DForecaster    — 1D-CNN baseline (faster, competitive on CMAPSS FD001)
  3. MCDropoutWrapper   — Wraps any model for uncertainty quantification

Monte Carlo Dropout (MC Dropout):
  - Keep dropout ACTIVE at inference time (model.train() selectively)
  - Run N forward passes → distribution of RUL predictions
  - Report: mean (point estimate) + std (uncertainty / confidence interval)
  - Reference: Gal & Ghahramani 2016, "Dropout as a Bayesian Approximation"
  - Why it matters for aerospace: safety-critical systems require uncertainty bounds
    on predictions, not just point estimates. This is what separates a research
    model from a PHM-deployable model.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_features:   int   = 15      # 14 CMAPSS sensors + health_index
    seq_len:      int   = 30
    hidden_dim:   int   = 128
    n_layers:     int   = 2
    dropout:      float = 0.3
    n_mc_samples: int   = 50      # MC Dropout forward passes


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM for time-series RUL regression.
    
    Architecture chosen to match aerospace PHM literature:
      - Stacked LSTM captures multi-scale temporal dependencies
      - Dropout between layers prevents overfitting on limited engine units
      - Single output neuron: scalar RUL (regression, not classification)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_dim)
        out = self.dropout(out[:, -1]) # last timestep: (batch, hidden_dim)
        return self.head(out).squeeze(-1)


class CNN1DForecaster(nn.Module):
    """
    1D Convolutional baseline — faster training, competitive on FD001.
    Good for benchmarking: if LSTM >> CNN, the temporal long-range dependencies
    matter (expected for FD002/FD004 with multiple operating conditions).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Conv1d(cfg.n_features, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → transpose for Conv1d
        x = x.transpose(1, 2)          # (batch, n_features, seq_len)
        x = self.encoder(x)             # (batch, 128, seq_len)
        x = self.pool(x).squeeze(-1)    # (batch, 128)
        return self.head(x).squeeze(-1)


class MCDropoutWrapper:
    """
    Wraps a trained model to enable MC Dropout inference.
    
    Usage:
        mc = MCDropoutWrapper(model, n_samples=50)
        mean_rul, std_rul, ci_lower, ci_upper = mc.predict(x)
    
    The 90% confidence interval (ci_lower, ci_upper) is what you show
    in dashboards and report in your README metrics table.
    """

    def __init__(self, model: nn.Module, n_samples: int = 50, ci: float = 0.90):
        self.model     = model
        self.n_samples = n_samples
        self.ci        = ci

    @torch.no_grad()
    def predict(self, x: torch.Tensor, device: str = "cpu"):
        """
        Args:
            x: Input tensor (batch, seq_len, n_features)
        Returns:
            mean_rul  : (batch,)
            std_rul   : (batch,)
            ci_lower  : (batch,)  lower bound of confidence interval
            ci_upper  : (batch,)  upper bound
        """
        self.model.train()   # keep dropout active
        x = x.to(device)

        samples = torch.stack(
            [self.model(x) for _ in range(self.n_samples)], dim=0
        )  # (n_samples, batch)

        mean_rul = samples.mean(0)
        std_rul  = samples.std(0)

        alpha     = (1 - self.ci) / 2
        ci_lower  = samples.quantile(alpha, dim=0)
        ci_upper  = samples.quantile(1 - alpha, dim=0)

        return (
            mean_rul.cpu().numpy(),
            std_rul.cpu().numpy(),
            ci_lower.cpu().numpy(),
            ci_upper.cpu().numpy(),
        )


def build_model(
    arch: str,
    cfg: ModelConfig,
    checkpoint_path: str = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Factory: instantiate and optionally load checkpoint.
    arch: "lstm" | "cnn1d"
    """
    arch_map = {"lstm": LSTMForecaster, "cnn1d": CNN1DForecaster}
    if arch not in arch_map:
        raise ValueError(f"arch must be one of {list(arch_map)}; got '{arch}'")

    model = arch_map[arch](cfg).to(device)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[build_model] Loaded checkpoint: {checkpoint_path}")

    return model
