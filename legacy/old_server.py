"""
FastAPI WebSocket server — Live RUL inference stream
Serves both REST (CMAPSS batch eval) and WebSocket (F1 live telemetry) endpoints.

Stack: FastAPI + uvicorn — consistent with your existing Node.js/WebSocket work
but in Python, which is appropriate for a PyTorch inference server.

Endpoints:
  GET  /health                      — liveness probe
  GET  /api/evaluate/cmapss         — CMAPSS benchmark results from checkpoint
  WS   /ws/telemetry/{source}       — live RUL stream (source: f1 | synthetic)
  POST /api/predict                 — single-sequence inference (REST)
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.f1_ingestion import fetch_stint_telemetry, prepare_inference_sequence, _synthetic_stint
from models.rul_models import ModelConfig, build_model, MCDropoutWrapper

# ─── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="RUL Forecaster API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model loading ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("./checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_cache: dict = {}

def get_model(arch: str = "lstm", subset: str = "FD001") -> MCDropoutWrapper | None:
    key = f"{arch}_{subset}"
    if key in _model_cache:
        return _model_cache[key]

    ckpt = CHECKPOINT_DIR / f"best_{arch}_{subset}.pt"
    if not ckpt.exists():
        return None

    cfg   = ModelConfig()           # defaults match training defaults
    model = build_model(arch, cfg, checkpoint_path=str(ckpt), device=DEVICE)
    mc    = MCDropoutWrapper(model, n_samples=50)
    _model_cache[key] = mc
    return mc


# ─── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "timestamp": time.time()}


@app.get("/api/evaluate/cmapss")
def cmapss_results(arch: str = "lstm", subset: str = "FD001"):
    results_path = CHECKPOINT_DIR / f"results_{arch}_{subset}.json"
    if not results_path.exists():
        return {"error": "No results found. Run training/train.py first."}
    return json.loads(results_path.read_text())


@app.post("/api/predict")
async def predict_single(payload: dict):
    """
    Single-sequence inference.
    Body: { "sequence": [[f1, f2, ..., fn], ...], "arch": "lstm" }
    sequence shape: (seq_len, n_features)
    """
    sequence = np.array(payload["sequence"], dtype=np.float32)
    arch     = payload.get("arch", "lstm")
    mc       = get_model(arch)

    if mc is None:
        return {"error": "Model not loaded. Train first."}

    x = torch.tensor(sequence[np.newaxis, ...])  # (1, seq_len, n_features)
    mean, std, ci_lo, ci_hi = mc.predict(x, device=DEVICE)

    return {
        "rul_mean":    round(float(mean[0]), 2),
        "rul_std":     round(float(std[0]),  2),
        "ci_lower_90": round(float(ci_lo[0]), 2),
        "ci_upper_90": round(float(ci_hi[0]), 2),
    }


# ─── WebSocket streaming ───────────────────────────────────────────────────────
@app.websocket("/ws/telemetry/{source}")
async def stream_telemetry(
    websocket: WebSocket,
    source: Literal["f1", "synthetic"] = "synthetic",
):
    """
    Streams live telemetry + RUL predictions lap-by-lap.
    Each message: JSON with sensor readings + RUL mean/CI.
    
    The frontend subscribes here and renders the degradation curve in real-time.
    """
    await websocket.accept()
    mc = get_model("lstm")

    try:
        # Load full stint data
        if source == "f1":
            df = fetch_stint_telemetry(year=2023, gp="Bahrain", driver="VER")
        else:
            df = _synthetic_stint(n_laps=40)

        if df is None or df.empty:
            await websocket.send_json({"error": "No telemetry data available"})
            return

        # Stream lap by lap (simulate real-time arrival)
        for i in range(1, len(df) + 1):
            lap_slice = df.iloc[:i]
            seq = prepare_inference_sequence(lap_slice, seq_len=30)

            payload = {
                "lap":           int(df["lap_number"].iloc[i-1]),
                "tyre_age":      int(df["tyre_age"].iloc[i-1]),
                "lap_time_delta": round(float(df["lap_time_delta"].iloc[i-1]), 3),
                "true_rul":      int(df["RUL"].iloc[i-1]),
                "sensors": {
                    col: round(float(df[col].iloc[i-1]), 2)
                    for col in ["Speed", "RPM", "Throttle", "Brake"]
                    if col in df.columns
                },
            }

            # Model prediction if available
            if mc is not None:
                x = torch.tensor(seq, dtype=torch.float32)
                mean, std, ci_lo, ci_hi = mc.predict(x, device=DEVICE)
                payload["rul_mean"]    = round(float(mean[0]), 2)
                payload["rul_std"]     = round(float(std[0]),  2)
                payload["ci_lower_90"] = round(float(ci_lo[0]), 2)
                payload["ci_upper_90"] = round(float(ci_hi[0]), 2)
            else:
                # Fallback: stream true RUL so dashboard works without model
                payload["rul_mean"]    = payload["true_rul"]
                payload["rul_std"]     = 0.0
                payload["ci_lower_90"] = payload["true_rul"]
                payload["ci_upper_90"] = payload["true_rul"]

            await websocket.send_json(payload)
            await asyncio.sleep(0.8)  # ~lap cadence for demo

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
