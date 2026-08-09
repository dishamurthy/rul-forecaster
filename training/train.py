"""
Training script — CMAPSS RUL Forecaster
Produces the benchmark numbers you put in your README and resume.

Standard CMAPSS FD001 LSTM targets from literature:
  RMSE ≈ 12–18 laps (varies by architecture + seq_len)
  MAE  ≈  9–14 laps
  Score (NASA metric) < 300

Run:
  python training/train.py --subset FD001 --arch lstm --epochs 100
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cmapss_loader import get_dataloaders
from models.rul_models import ModelConfig, build_model, MCDropoutWrapper


# ─── NASA Score metric ────────────────────────────────────────────────────────
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Asymmetric scoring function penalising late predictions more than early.
    Lower is better. Published in Saxena et al. 2008 (the CMAPSS paper).
    Cite this in your README.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(scores.sum())


# ─── Training loop ─────────────────────────────────────────────────────────────
def train(cfg: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device} | Subset: {cfg.subset} | Arch: {cfg.arch}")

    train_loader, test_loader, n_features = get_dataloaders(
        cfg.data_dir, cfg.subset, cfg.seq_len, cfg.batch_size
    )

    model_cfg = ModelConfig(
        n_features=n_features,
        seq_len=cfg.seq_len,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    model = build_model(cfg.arch, model_cfg, device=device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    history   = []

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── Evaluate ──
        if epoch % cfg.eval_every == 0:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    preds.extend(model(x).cpu().numpy())
                    targets.extend(y.numpy())

            preds   = np.array(preds)
            targets = np.array(targets)
            rmse    = float(np.sqrt(np.mean((preds - targets) ** 2)))
            mae     = float(np.mean(np.abs(preds - targets)))
            score   = nasa_score(targets, preds)
            train_loss = np.mean(train_losses)

            print(
                f"Epoch {epoch:>4}/{cfg.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | Score: {score:.1f}"
            )

            history.append({"epoch": epoch, "loss": train_loss,
                            "rmse": rmse, "mae": mae, "score": score})

            if rmse < best_rmse:
                best_rmse = rmse
                out_dir = Path(cfg.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ckpt = out_dir / f"best_{cfg.arch}_{cfg.subset}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"  ✓ Saved checkpoint → {ckpt} (RMSE {rmse:.2f})")

    # ── Final MC Dropout evaluation ──
    best_model = build_model(
        cfg.arch, model_cfg,
        checkpoint_path=str(Path(cfg.out_dir) / f"best_{cfg.arch}_{cfg.subset}.pt"),
        device=device,
    )
    mc = MCDropoutWrapper(best_model, n_samples=50)

    all_x = torch.cat([x for x, _ in test_loader])
    all_y = torch.cat([y for _, y in test_loader]).numpy()

    mean_rul, std_rul, ci_lo, ci_hi = mc.predict(all_x, device=device)
    mc_rmse  = float(np.sqrt(np.mean((mean_rul - all_y) ** 2)))
    mc_mae   = float(np.mean(np.abs(mean_rul - all_y)))
    mc_score = nasa_score(all_y, mean_rul)
    # Calibration: what fraction of true RUL falls within predicted 90% CI?
    coverage = float(np.mean((all_y >= ci_lo) & (all_y <= ci_hi)))

    results = {
        "subset":    cfg.subset,
        "arch":      cfg.arch,
        "best_rmse": round(mc_rmse, 2),
        "best_mae":  round(mc_mae, 2),
        "nasa_score": round(mc_score, 1),
        "ci_coverage_90pct": round(coverage, 3),
        "mean_uncertainty_std": round(float(std_rul.mean()), 2),
        "history": history,
    }

    results_path = Path(cfg.out_dir) / f"results_{cfg.arch}_{cfg.subset}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n── Final Results ({cfg.arch} / {cfg.subset}) ──")
    print(f"  RMSE      : {mc_rmse:.2f}")
    print(f"  MAE       : {mc_mae:.2f}")
    print(f"  NASA Score: {mc_score:.1f}")
    print(f"  90% CI coverage: {coverage:.1%}  (target ≥ 0.90)")
    print(f"  Results saved → {results_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/raw/CMAPSS")
    p.add_argument("--out_dir",    default="./checkpoints")
    p.add_argument("--subset",     default="FD001", choices=["FD001","FD002","FD003","FD004"])
    p.add_argument("--arch",       default="lstm",  choices=["lstm","cnn1d"])
    p.add_argument("--seq_len",    type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers",   type=int, default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=5)
    cfg = p.parse_args()
    train(cfg)
