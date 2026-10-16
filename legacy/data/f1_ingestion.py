"""
FastF1 Telemetry Ingestion — Live Inference Data Source
Pulls lap-by-lap sensor data as a proxy for high-frequency aerospace telemetry.

Why F1 telemetry?
- Sampling rate: ~18Hz (240 samples/lap @ ~13s/lap avg)
- Multi-channel sensor streams identical in structure to aerospace PHM data
- Ground truth degradation: tyre compound age + lap time delta as failure proxy
- Accessible, documented, reproducible — no NDA required

Channel mapping to aerospace analogy:
  Speed       → Engine airspeed / Mach sensor
  RPM         → Shaft rotation speed (N1/N2 in turbofan)
  Throttle    → Fuel flow / thrust command
  Brake       → Actuator load / stress cycle
  nGear       → Operating regime index
  DRS         → Drag state (binary mode flag)
  Tyre age    → Component cycle count (hours-on-wing equivalent)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import fastf1
    fastf1.Cache.enable_cache("./cache/f1")
    HAS_F1 = True
except ImportError:
    HAS_F1 = False
    print("[f1_ingestion] fastf1 not installed. Run: pip install fastf1")


TELEMETRY_CHANNELS = ["Speed", "RPM", "Throttle", "Brake", "nGear", "DRS"]


def fetch_stint_telemetry(
    year: int,
    gp: str,
    session_type: str = "R",
    driver: str = "VER",
    compound: str = "MEDIUM",
) -> pd.DataFrame | None:
    """
    Fetch telemetry for a single stint (one tyre compound run).
    Returns a DataFrame with time-series sensor data + tyre age + lap delta.
    
    Args:
        year: Season year (e.g. 2023)
        gp: Grand Prix name or round number (e.g. "Bahrain" or 1)
        session_type: "R" for Race, "Q" for Qualifying
        driver: Three-letter driver code
        compound: Tyre compound to filter for ("SOFT", "MEDIUM", "HARD")
    
    Returns:
        DataFrame with columns: lap_number, tyre_age, lap_time_delta, + telemetry channels
        None if data unavailable.
    """
    if not HAS_F1:
        return _synthetic_stint(n_laps=35)

    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception as e:
        print(f"[f1_ingestion] Session load failed: {e}. Using synthetic data.")
        return _synthetic_stint(n_laps=35)

    laps = session.laps.pick_driver(driver).pick_compound(compound).reset_index(drop=True)
    if laps.empty:
        return None

    records = []
    baseline_lap_time = None

    for _, lap in laps.iterrows():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue

        if tel.empty:
            continue

        # Resample to fixed 50-point summary per lap (mean of each channel)
        summary = {ch: float(tel[ch].mean()) for ch in TELEMETRY_CHANNELS if ch in tel.columns}
        summary["lap_number"] = int(lap["LapNumber"]) if not pd.isna(lap["LapNumber"]) else -1
        summary["tyre_age"]   = int(lap["TyreLife"])  if not pd.isna(lap["TyreLife"])  else 0

        lap_time_s = lap["LapTime"].total_seconds() if not pd.isna(lap["LapTime"]) else None
        if lap_time_s:
            if baseline_lap_time is None:
                baseline_lap_time = lap_time_s
            summary["lap_time_delta"] = lap_time_s - baseline_lap_time
        else:
            summary["lap_time_delta"] = 0.0

        records.append(summary)

    if not records:
        return None

    df = pd.DataFrame(records)
    df = _add_rul_labels(df)
    return df


def _add_rul_labels(df: pd.DataFrame, threshold_delta: float = 1.5) -> pd.DataFrame:
    """
    Construct RUL labels: laps remaining before lap_time_delta exceeds threshold.
    threshold_delta: seconds above baseline considered 'degraded'
    """
    df = df.copy()
    n = len(df)
    rul = []

    # Find failure point: first lap where delta > threshold (sustained)
    failure_idx = n  # default: never fails
    for i in range(n):
        if df["lap_time_delta"].iloc[i] > threshold_delta:
            failure_idx = i
            break

    for i in range(n):
        rul.append(max(0, failure_idx - i))

    df["RUL"] = rul
    return df


def build_f1_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Health Index for F1 stint data.
    Incorporates tyre age as a hard prior (like hours-on-wing in aerospace).
    """
    df = df.copy()
    sensor_cols = [c for c in TELEMETRY_CHANNELS if c in df.columns]

    scaler = MinMaxScaler()
    normed = scaler.fit_transform(df[sensor_cols])
    normed_df = pd.DataFrame(normed, columns=sensor_cols, index=df.index)

    # Weight RPM and Speed higher — best degradation correlates
    weights = pd.Series({
        "Speed": 0.20, "RPM": 0.25, "Throttle": 0.15,
        "Brake": 0.15, "nGear": 0.10, "DRS": 0.05,
    })
    w = weights.reindex(sensor_cols).fillna(0.05)
    w = w / w.sum()

    # Tyre age penalty: normalise to [0, 1] and invert (old tyre → lower HI)
    max_age = df["tyre_age"].max() or 1
    tyre_penalty = 1 - (df["tyre_age"] / max_age) * 0.3  # 30% weight on age

    df["health_index"] = (normed_df * w).sum(axis=1) * tyre_penalty
    df["health_index"] = MinMaxScaler().fit_transform(df[["health_index"]])
    return df


def _synthetic_stint(n_laps: int = 35, seed: int = 42) -> pd.DataFrame:
    """
    Generates plausible synthetic stint data when FastF1 unavailable.
    Used for testing and CI. NOT for training — use CMAPSS for that.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_laps)

    # Simulate degradation curve: gradual decline after lap 20
    degradation = np.where(t > 20, (t - 20) * 0.08, 0.0)

    df = pd.DataFrame({
        "lap_number":      t + 1,
        "tyre_age":        t,
        "Speed":           300 - degradation * 2  + rng.normal(0, 2, n_laps),
        "RPM":             11500 - degradation * 50 + rng.normal(0, 100, n_laps),
        "Throttle":        0.82 - degradation * 0.005 + rng.normal(0, 0.02, n_laps),
        "Brake":           0.15 + degradation * 0.003 + rng.normal(0, 0.01, n_laps),
        "nGear":           6.5 + rng.normal(0, 0.3, n_laps),
        "DRS":             rng.integers(0, 2, n_laps).astype(float),
        "lap_time_delta":  degradation + rng.normal(0, 0.1, n_laps),
    })

    df = _add_rul_labels(df)
    return df


def prepare_inference_sequence(
    df: pd.DataFrame,
    seq_len: int = 30,
    scaler=None,
) -> np.ndarray:
    """
    Prepare the latest seq_len laps as input to the model.
    Returns array of shape (1, seq_len, n_features).
    """
    df = build_f1_health_index(df)
    feature_cols = [c for c in TELEMETRY_CHANNELS if c in df.columns] + ["health_index"]

    vals = df[feature_cols].values.astype(np.float32)

    if scaler is not None:
        vals = scaler.transform(vals)

    if len(vals) >= seq_len:
        seq = vals[-seq_len:]
    else:
        pad = np.zeros((seq_len - len(vals), vals.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, vals])

    return seq[np.newaxis, ...]  # (1, seq_len, n_features)
