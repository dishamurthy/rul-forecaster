"""
CMAPSS (NASA C-MAPSS) Turbofan Engine Degradation Dataset Loader
Dataset: FD001–FD004 from NASA Prognostics Data Repository
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

This is the standard aerospace PHM benchmark. Cite this in your README and interviews.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Column definitions per NASA documentation
INDEX_COLS = ["unit_id", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors recommended by literature (drop near-constant ones: s1,s5,s6,s10,s16,s18,s19)
ACTIVE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9",
                  "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

# Piecewise linear RUL cap — standard in PHM literature
RUL_CAP = 125


def load_cmapss(data_dir: str, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CMAPSS train/test split for a given subset.
    
    Args:
        data_dir: Path to directory containing train_FD001.txt etc.
        subset: One of FD001, FD002, FD003, FD004
    
    Returns:
        (train_df, test_df) with RUL column added to train
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"train_{subset}.txt"
    test_path  = data_dir / f"test_{subset}.txt"
    rul_path   = data_dir / f"RUL_{subset}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLS)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=ALL_COLS)
    rul_true = pd.read_csv(rul_path,   sep=r"\s+", header=None, names=["RUL"])

    # Construct RUL labels for training data (piecewise linear, capped)
    max_cycles = train_df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="unit_id")
    train_df["RUL"] = (train_df["max_cycle"] - train_df["cycle"]).clip(upper=RUL_CAP)
    train_df.drop("max_cycle", axis=1, inplace=True)

    # Attach true RUL to test sequences (last cycle of each unit)
    test_df = test_df.merge(
        rul_true.assign(unit_id=range(1, len(rul_true)+1)),
        on="unit_id", how="left"
    )

    return train_df, test_df


def build_health_index(df: pd.DataFrame, sensors: list[str] = ACTIVE_SENSORS) -> pd.DataFrame:
    """
    Construct a composite Health Index (HI) from multi-sensor signals.
    
    HI is a dimensionality-reduced scalar in [0,1] where:
      1.0 = healthy, 0.0 = near failure
    
    Method: weighted average of per-sensor min-max normalisation,
    where weights are the absolute Pearson correlation with RUL.
    Mirrors NASA PHM framework HI construction.
    """
    df = df.copy()
    scaler = MinMaxScaler()

    sensor_data = df[sensors].values
    normalised  = scaler.fit_transform(sensor_data)
    norm_df     = pd.DataFrame(normalised, columns=sensors, index=df.index)

    if "RUL" in df.columns:
        corr = norm_df.corrwith(df["RUL"]).abs()
        weights = corr / corr.sum()
        df["health_index"] = (norm_df * weights).sum(axis=1)
    else:
        df["health_index"] = norm_df.mean(axis=1)

    return df


class CMAPSSDataset(Dataset):
    """
    Sliding-window sequence dataset for CMAPSS.
    Each sample: (seq_len, n_features) → scalar RUL
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 30,
        sensors: list[str] = ACTIVE_SENSORS,
        include_hi: bool = True,
        mode: str = "train",          # "train" or "test"
    ):
        self.seq_len    = seq_len
        self.sensors    = sensors
        self.include_hi = include_hi
        self.mode       = mode

        df = build_health_index(df, sensors)
        feature_cols = sensors + (["health_index"] if include_hi else [])
        self.feature_cols = feature_cols

        # Normalise per-unit to remove operational condition bias
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        self.scaler = scaler

        self.sequences, self.labels = self._make_sequences(df)

    def _make_sequences(self, df: pd.DataFrame):
        seqs, labels = [], []
        for uid, group in df.groupby("unit_id"):
            vals = group[self.feature_cols].values.astype(np.float32)
            rul  = group["RUL"].values.astype(np.float32)

            if self.mode == "train":
                for i in range(len(vals) - self.seq_len + 1):
                    seqs.append(vals[i : i + self.seq_len])
                    labels.append(rul[i + self.seq_len - 1])
            else:
                # Test: use last seq_len cycles only
                if len(vals) >= self.seq_len:
                    seqs.append(vals[-self.seq_len:])
                else:
                    pad = np.zeros((self.seq_len - len(vals), vals.shape[1]), dtype=np.float32)
                    seqs.append(np.vstack([pad, vals]))
                labels.append(rul.iloc[-1] if hasattr(rul, 'iloc') else rul[-1])

        return (
            torch.tensor(np.array(seqs), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32),
        )

    def __len__(self):  return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]


def get_dataloaders(
    data_dir: str,
    subset: str = "FD001",
    seq_len: int = 30,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Returns (train_loader, test_loader, n_features)
    """
    train_df, test_df = load_cmapss(data_dir, subset)

    train_ds = CMAPSSDataset(train_df, seq_len=seq_len, mode="train")
    test_ds  = CMAPSSDataset(test_df,  seq_len=seq_len, mode="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, len(train_ds.feature_cols)
