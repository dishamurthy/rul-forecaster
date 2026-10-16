"""
RUL Forecaster — FastAPI server + dashboard
===========================================
Serves the dashboard UI and real PyTorch inference on the NASA C-MAPSS FD001 test engines.

Run (from the project root):
    uvicorn api.server:app --reload --port 8000
Then open http://127.0.0.1:8000

Endpoints
    GET  /                          dashboard UI
    GET  /docs                      interactive Swagger UI (auto-generated)
    GET  /api/status                what is loaded: data files, checkpoints, device
    GET  /api/benchmark             RMSE / MAE / NASA score / CI coverage for both models (computed live)
    GET  /api/engine/{id}           full MC-Dropout trajectory for one test engine
    POST /api/predict               single 30x15 window -> RUL mean / std / 90% CI
    WS   /ws/replay/{id}            streams the trajectory cycle-by-cycle (live-telemetry replay)
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api import nets, pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
CKPT_DIRS = [ROOT / "checkpoints", ROOT]
STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = ROOT / "assets"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MC_SAMPLES = 50
MC_SEED = 1234          # fixed so the dashboard shows the same curve every run

app = FastAPI(
    title="RUL Forecaster API",
    version="2.0.0",
    description="Remaining-useful-life prediction for turbofan engines (NASA C-MAPSS FD001) "
                "with Monte-Carlo-Dropout uncertainty. LSTM+Transformer and 1D-CNN models.",
)


# ─── Lazy singletons ────────────────────────────────────────────────────────────
_lock = threading.RLock()          # also guards the global torch RNG seed
_pipe: Optional[pipeline.CMAPSSPipeline] = None
_models: Dict[str, torch.nn.Module] = {}
_traj_cache: Dict[Tuple[str, int, int], dict] = {}
_bench_cache: Optional[dict] = None


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


@app.exception_handler(ApiError)
async def _api_error_handler(_, exc: ApiError):
    return JSONResponse(status_code=exc.status, content={"code": exc.code, "message": exc.message})


def get_pipeline() -> pipeline.CMAPSSPipeline:
    global _pipe
    with _lock:
        if _pipe is None:
            missing = pipeline.missing_data_files(DATA_DIR)
            if missing:
                raise ApiError(
                    503, "DATA_MISSING",
                    f"C-MAPSS files not found in data/raw/: {', '.join(missing)}. "
                    "Run:  python scripts/download_data.py")
            _pipe = pipeline.CMAPSSPipeline(DATA_DIR)
        return _pipe


def get_model(arch: str) -> torch.nn.Module:
    if arch not in nets.ARCHS:
        raise ApiError(400, "BAD_ARCH", f"arch must be one of {list(nets.ARCHS)}")
    with _lock:
        if arch not in _models:
            ckpt = nets.find_checkpoint(arch, CKPT_DIRS)
            if ckpt is None:
                raise ApiError(503, "CHECKPOINT_MISSING",
                               f"{nets.ARCHS[arch][1]} not found in checkpoints/.")
            _models[arch] = nets.load_model(arch, ckpt, DEVICE)
        return _models[arch]


def _check_engine(pipe: pipeline.CMAPSSPipeline, engine_id: int):
    if engine_id not in pipe.engines:
        raise ApiError(404, "NO_SUCH_ENGINE", f"engine_id must be 1..{pipe.n_engines}")


# ─── Core computations (blocking — always called via run_in_threadpool) ─────────
def compute_trajectory(arch: str, engine_id: int, n_samples: int) -> dict:
    key = (arch, engine_id, n_samples)
    with _lock:
        if key in _traj_cache:
            return _traj_cache[key]
        pipe, model = get_pipeline(), get_model(arch)
        _check_engine(pipe, engine_id)

        X, true_rul, end_idx = pipe.trajectory_windows(engine_id)
        torch.manual_seed(MC_SEED)
        mean, std, lo, hi = nets.predict_mc(model, X, DEVICE, n_samples=n_samples)
        cap = float(pipeline.RUL_CAP)
        e = pipe.engine(engine_id)

        traj = {
            "engine_id": engine_id,
            "arch": arch,
            "n_cycles": int(len(e["cycles"])),
            "final_rul": e["final_rul"],
            "cycles": e["cycles"][end_idx],
            "true_rul": true_rul,
            "mean": np.clip(mean, 0, cap),
            "std": std,
            "lo": np.clip(lo, 0, cap),
            "hi": np.clip(hi, 0, cap),
            "health_index": e["features"][end_idx, -1],
            "raw": e["raw"][end_idx],
            "norm": e["norm"][end_idx],
        }
        _traj_cache[key] = traj
        return traj


def step_payload(t: dict, i: int) -> dict:
    return {
        "cycle": int(t["cycles"][i]),
        "true_rul": round(float(t["true_rul"][i]), 2),
        "rul_mean": round(float(t["mean"][i]), 2),
        "rul_std": round(float(t["std"][i]), 2),
        "ci_lo": round(float(t["lo"][i]), 2),
        "ci_hi": round(float(t["hi"][i]), 2),
        "health_index": round(float(t["health_index"][i]), 4),
        "sensors": {
            s: {"raw": round(float(t["raw"][i, j]), 3), "norm": round(float(t["norm"][i, j]), 4)}
            for j, s in enumerate(pipeline.ACTIVE_SENSORS)
        },
    }


def compute_benchmark() -> dict:
    """Official test metric: last 30 cycles of each of the 100 test engines, capped labels."""
    global _bench_cache
    with _lock:
        if _bench_cache is not None:
            return _bench_cache
        pipe = get_pipeline()
        uids, X, y = pipe.final_windows()
        out = {"models": {}, "fleet": {}, "n_engines": len(uids), "rul_cap": pipeline.RUL_CAP}

        for arch in nets.ARCHS:
            try:
                model = get_model(arch)
            except ApiError:
                continue
            pred = nets.predict_point(model, X, DEVICE)                      # eval mode = notebook metric
            rmse, mae = pipeline.rmse_mae(y, pred)
            torch.manual_seed(MC_SEED)
            _, _, lo, hi = nets.predict_mc(model, X, DEVICE, n_samples=DEFAULT_MC_SAMPLES)
            coverage = float(np.mean((y >= lo) & (y <= hi)))
            out["models"][arch] = {
                "name": nets.ARCHS[arch][2],
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "nasa_score": round(pipeline.nasa_score(y, pred), 1),
                "ci90_coverage": round(coverage, 3),
                "n_params": int(sum(p.numel() for p in model.parameters())),
            }
            out["fleet"][arch] = [
                {"engine": int(u), "cycles": int(len(pipe.engine(u)["cycles"])),
                 "true_rul": round(float(t), 1), "pred_rul": round(float(p), 1)}
                for u, t, p in zip(uids, y, pred)
            ]
        if not out["models"]:
            raise ApiError(503, "CHECKPOINT_MISSING", "No model checkpoints found in checkpoints/.")
        _bench_cache = out
        return out


# ─── REST ───────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    missing = pipeline.missing_data_files(DATA_DIR)
    return {
        "status": "ok",
        "device": DEVICE,
        "torch": torch.__version__,
        "data_ready": not missing,
        "data_missing": missing,
        "checkpoints": {a: nets.find_checkpoint(a, CKPT_DIRS) is not None for a in nets.ARCHS},
        "models": {a: v[2] for a, v in nets.ARCHS.items()},
        "mc_samples": DEFAULT_MC_SAMPLES,
        "timestamp": time.time(),
    }


@app.get("/api/benchmark")
async def benchmark():
    return await run_in_threadpool(compute_benchmark)


@app.get("/api/engine/{engine_id}")
async def engine_trajectory(engine_id: int, arch: str = "lstm", mc_samples: int = DEFAULT_MC_SAMPLES):
    """Full run for one test engine: per-cycle RUL mean, std, 90% CI, health index, sensors."""
    mc_samples = max(5, min(200, mc_samples))
    t = await run_in_threadpool(compute_trajectory, arch, engine_id, mc_samples)
    return {
        "engine_id": engine_id, "arch": arch, "n_cycles": t["n_cycles"],
        "final_rul": t["final_rul"], "mc_samples": mc_samples,
        "steps": [step_payload(t, i) for i in range(len(t["cycles"]))],
    }


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="30 cycles x 15 features (14 MinMax-scaled sensors + health index)")
    arch: str = "lstm"
    mc_samples: int = Field(DEFAULT_MC_SAMPLES, ge=5, le=200)


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Single-window inference with MC-Dropout uncertainty."""
    arr = np.asarray(req.sequence, dtype=np.float32)
    if arr.shape != (pipeline.SEQ_LEN, len(pipeline.FEATURE_COLS)):
        raise HTTPException(
            422, f"sequence must have shape ({pipeline.SEQ_LEN}, {len(pipeline.FEATURE_COLS)}); got {tuple(arr.shape)}")

    def _run():
        with _lock:
            model = get_model(req.arch)
            torch.manual_seed(MC_SEED)
            return nets.predict_mc(model, arr[None, ...], DEVICE, n_samples=req.mc_samples)

    mean, std, lo, hi = await run_in_threadpool(_run)
    cap = float(pipeline.RUL_CAP)
    return {
        "arch": req.arch,
        "rul_mean": round(float(np.clip(mean[0], 0, cap)), 2),
        "rul_std": round(float(std[0]), 2),
        "ci_lower_90": round(float(np.clip(lo[0], 0, cap)), 2),
        "ci_upper_90": round(float(np.clip(hi[0], 0, cap)), 2),
    }


# ─── WebSocket: live replay ─────────────────────────────────────────────────────
@app.websocket("/ws/replay/{engine_id}")
async def ws_replay(ws: WebSocket, engine_id: int, arch: str = "lstm",
                    interval: float = 0.12, start: Optional[int] = None,
                    mc_samples: int = DEFAULT_MC_SAMPLES):
    """
    Streams one engine's run cycle-by-cycle, as if sensor data were arriving live.
    Messages:  {"type":"init",...}  ->  {"type":"step",...} x N  ->  {"type":"done"}
    `start` = cycle to resume from (the UI uses this for pause/resume and speed changes).
    """
    await ws.accept()
    interval = max(0.01, min(2.0, interval))
    mc_samples = max(5, min(200, mc_samples))
    try:
        try:
            t = await run_in_threadpool(compute_trajectory, arch, engine_id, mc_samples)
        except ApiError as e:
            await ws.send_json({"type": "error", "code": e.code, "message": e.message})
            await ws.close()
            return

        cycles = t["cycles"]
        first = 0 if start is None else int(np.searchsorted(cycles, start))
        await ws.send_json({
            "type": "init", "engine_id": engine_id, "arch": arch,
            "n_cycles": t["n_cycles"], "first_cycle": int(cycles[0]), "last_cycle": int(cycles[-1]),
            "final_rul": t["final_rul"], "mc_samples": mc_samples,
        })
        for i in range(first, len(cycles)):
            await ws.send_json({"type": "step", **step_payload(t, i)})
            await asyncio.sleep(interval)
        await ws.send_json({"type": "done"})
        await ws.close()
    except WebSocketDisconnect:
        pass


# ─── Static files / UI ──────────────────────────────────────────────────────────
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
