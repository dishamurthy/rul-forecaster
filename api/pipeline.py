"""
Data pipeline for the dashboard — mirrors the notebook (Cells 3-5) exactly so the
saved checkpoints see the same input distribution they were trained on.

  * 14 active sensors, MinMax-scaled with a scaler fit on the FD001 *training* set
  * 15th feature = health index (see note below)
  * 30-cycle sliding windows; test RUL labels capped at 125

Pure numpy / pandas / scikit-learn (no torch) so it can be unit-tested on its own.

Note on the health index: in the notebook, the training set uses a correlation-weighted
HI but the test set has no RUL column, so it falls back to the plain sensor mean.
We replicate that behaviour on purpose — it is what the checkpoints were evaluated with,
and it is what reproduces the numbers in the README.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RUL_CAP = 125
SEQ_LEN = 30
SUBSET = "FD001"

ACTIVE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
                  "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
FEATURE_COLS = ACTIVE_SENSORS + ["health_index"]          # 15 features

_INDEX = ["unit_id", "cycle"]
_SETTINGS = ["setting_1", "setting_2", "setting_3"]
_SENSORS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = _INDEX + _SETTINGS + _SENSORS

REQUIRED_FILES = [f"train_{SUBSET}.txt", f"test_{SUBSET}.txt", f"RUL_{SUBSET}.txt"]

# Human-readable names for the sensor panel (NASA C-MAPSS documentation)
SENSOR_LABELS = {
    "s2": "T24 · LPC outlet temp",
    "s3": "T30 · HPC outlet temp",
    "s4": "T50 · LPT outlet temp",
    "s7": "P30 · HPC outlet pressure",
    "s8": "Nf · fan speed",
    "s9": "Nc · core speed",
    "s11": "Ps30 · HPC static pressure",
    "s12": "phi · fuel flow ratio",
    "s13": "NRf · corrected fan speed",
    "s14": "NRc · corrected core speed",
    "s15": "BPR · bypass ratio",
    "s17": "htBleed · bleed enthalpy",
    "s20": "W31 · HPT coolant bleed",
    "s21": "W32 · LPT coolant bleed",
}


def missing_data_files(data_dir: Path) -> List[str]:
    return [f for f in REQUIRED_FILES if not (Path(data_dir) / f).exists()]


class CMAPSSPipeline:
    """Loads FD001 once and exposes per-engine feature matrices for the 100 test engines."""

    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        read = lambda name: pd.read_csv(data_dir / name, sep=r"\s+", header=None)

        train_df = read(f"train_{SUBSET}.txt")
        test_df = read(f"test_{SUBSET}.txt")
        rul_true = read(f"RUL_{SUBSET}.txt")

        train_df = train_df.iloc[:, :len(ALL_COLS)]
        test_df = test_df.iloc[:, :len(ALL_COLS)]
        train_df.columns = ALL_COLS
        test_df.columns = ALL_COLS

        # Scaler is fit on the training sensors only (as in the notebook)
        self.scaler = MinMaxScaler().fit(train_df[ACTIVE_SENSORS].values)

        # Test labels: one RUL per engine, capped at 125 (the notebook's "FIX")
        final_rul = rul_true.iloc[:, 0].clip(upper=RUL_CAP).to_numpy(dtype=np.float32)

        self.engines: Dict[int, dict] = {}
        for uid, g in test_df.groupby("unit_id"):
            g = g.sort_values("cycle")
            norm = self.scaler.transform(g[ACTIVE_SENSORS].values).astype(np.float32)
            hi = norm.mean(axis=1, keepdims=True)                 # test-time HI (see module note)
            feats = np.hstack([norm, hi]).astype(np.float32)      # (n_cycles, 15)
            self.engines[int(uid)] = {
                "cycles": g["cycle"].to_numpy(dtype=int),
                "features": feats,
                "norm": norm,
                "raw": g[ACTIVE_SENSORS].to_numpy(dtype=np.float32),
                "final_rul": float(final_rul[int(uid) - 1]),
            }

        self.n_engines = len(self.engines)

    # ── helpers ────────────────────────────────────────────────────────────────
    def engine(self, uid: int) -> dict:
        if uid not in self.engines:
            raise KeyError(uid)
        return self.engines[uid]

    def final_windows(self):
        """The 100 last-30-cycle windows + labels used for the official test metric."""
        uids = sorted(self.engines)
        X, y = [], []
        for uid in uids:
            e = self.engines[uid]
            X.append(_last_window(e["features"]))
            y.append(e["final_rul"])
        return uids, np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32)

    def trajectory_windows(self, uid: int):
        """
        Every full 30-cycle window of one test engine (cycle 30 .. n) with its true RUL.
        True RUL at cycle t = final_rul + (n - t), capped at 125.
        """
        e = self.engine(uid)
        feats, cycles = e["features"], e["cycles"]
        n = len(feats)
        idx = list(range(SEQ_LEN, n + 1))                          # window ends at cycle index t-1
        X = np.stack([feats[t - SEQ_LEN:t] for t in idx]).astype(np.float32)
        end = np.asarray(idx) - 1
        true_rul = np.minimum(e["final_rul"] + (n - np.asarray(idx)), RUL_CAP).astype(np.float32)
        return X, true_rul, end


def _last_window(feats: np.ndarray) -> np.ndarray:
    """Last SEQ_LEN rows; left-pad with zeros if the engine is shorter (as in the notebook)."""
    if len(feats) >= SEQ_LEN:
        return feats[-SEQ_LEN:]
    pad = np.zeros((SEQ_LEN - len(feats), feats.shape[1]), np.float32)
    return np.vstack([pad, feats])


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM score from Saxena et al. 2008 — late predictions are penalised harder."""
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))
