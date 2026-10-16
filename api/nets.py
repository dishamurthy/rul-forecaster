"""
Model definitions + inference helpers.

The two classes below are copied verbatim from the notebook (Cell 6) — they are the
architectures that actually produced lstm_rul.pth / cnn_rul.pth, so state-dict keys match.
(`legacy/models/rul_models.py` is an earlier draft with a different architecture and will NOT
load these checkpoints.)

Monte Carlo Dropout (Gal & Ghahramani, 2016): keep dropout active at inference, run N stochastic
forward passes, report mean ± std and a 90% interval from the sample quantiles.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_features: int = 15
    seq_len: int = 30
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.2


class LSTMForecaster(nn.Module):
    """LSTM -> Transformer attention -> regression head"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=4,
            dim_feedforward=256, dropout=cfg.dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False only silences a warning; it does not change the weights
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.transformer(out)
        return self.head(out.mean(dim=1)).squeeze(-1)


class CNN1DForecaster(nn.Module):
    """1D CNN -> global avg pool -> regression head"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(cfg.n_features, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, cfg.hidden_dim, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.encoder(x.permute(0, 2, 1))
        return self.head(out.mean(dim=-1)).squeeze(-1)


# arch key -> (class, checkpoint filename, display name)
ARCHS = {
    "lstm": (LSTMForecaster, "lstm_rul.pth", "LSTM + Transformer attention"),
    "cnn": (CNN1DForecaster, "cnn_rul.pth", "1D-CNN baseline"),
}


def find_checkpoint(arch: str, search_dirs) -> "Path | None":
    fname = ARCHS[arch][1]
    for d in search_dirs:
        p = Path(d) / fname
        if p.exists():
            return p
    return None


def load_model(arch: str, ckpt_path: Path, device: str) -> nn.Module:
    cls = ARCHS[arch][0]
    model = cls(ModelConfig()).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_point(model: nn.Module, x: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    """Deterministic prediction (dropout off) — this is what the notebook's test metric uses."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_mc(model: nn.Module, x: np.ndarray, device: str,
               n_samples: int = 50, ci: float = 0.90, batch: int = 512):
    """
    MC-Dropout prediction.
    Returns mean, std, ci_lower, ci_upper — each shape (len(x),), clipped to the valid RUL range.
    """
    alpha = (1.0 - ci) / 2.0
    means, stds, los, his = [], [], [], []
    model.train()                       # dropout ON (no BatchNorm in these models, so this is safe)
    try:
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            s = torch.stack([model(xb) for _ in range(n_samples)], dim=0)   # (n_samples, b)
            means.append(s.mean(0).cpu().numpy())
            stds.append(s.std(0).cpu().numpy())
            los.append(s.quantile(alpha, dim=0).cpu().numpy())
            his.append(s.quantile(1 - alpha, dim=0).cpu().numpy())
    finally:
        model.eval()
    cat = np.concatenate
    return cat(means), cat(stds), cat(los), cat(his)
