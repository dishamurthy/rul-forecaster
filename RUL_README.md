# Remaining Useful Life (RUL) Predictor
### Deep Learning · Predictive Maintenance · NASA C-MAPSS Benchmark

---

## What this is

This project predicts how many operational cycles are left before a jet engine fails, using sensor readings from the last 30 cycles of its life. It's a classic **Prognostics and Health Management (PHM)** problem — the kind of thing aerospace, manufacturing, and EV battery companies actually use to decide when to pull equipment for maintenance before it breaks in production.

I built this partly because I wanted to understand time-series deep learning beyond the usual NLP/LLM work, and partly because the C-MAPSS dataset is the standard benchmark everyone in this space references. If you claim to do predictive maintenance, this is what you benchmark on.

---

## Dataset

**NASA C-MAPSS — Turbofan Engine Degradation Simulation (FD001 subset)**

NASA ran 100 turbofan engines to failure in simulation, recording 21 sensor channels every operational cycle. The training set contains the full run-to-failure history. The test set gives you a truncated history and asks you to predict how many cycles remain.

- 100 training engines, 20,631 total rows
- 100 test engines (one prediction per engine)
- 21 raw sensors → 14 informative ones selected (dropped near-constant channels)
- RUL capped at 125 cycles — standard piecewise linear labeling from Saxena et al. 2008

One non-obvious thing: the raw `RUL_FD001.txt` test labels are uncapped and go above 125. If you don't cap the test labels to match the training distribution, your test RMSE stays stuck around 32–38 no matter how long you train. That was the main bug I had to debug and fix in the data pipeline.

Reference: Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM Conference.

---

## Architecture

**LSTM + Transformer attention head**

Input: sliding window of 30 cycles × 15 features (14 sensors + composite Health Index)

```
Input (30 × 15)
    │
    ▼
2-layer LSTM (hidden=128, dropout=0.2)
    │  captures temporal dependencies across cycles
    ▼
Transformer Encoder (1 layer, 4 heads)
    │  attention over the LSTM output sequence
    ▼
Mean pooling → Linear(128→64) → ReLU → Dropout → Linear(64→1)
    │
    ▼
Scalar RUL prediction
```

**Why LSTM and not just a Transformer?**
LSTMs naturally handle variable-length sequences and are good at remembering slow-moving degradation trends that build over 50–100 cycles. The Transformer layer on top lets the model attend to which specific timesteps in the window matter most — early cycles in the window might be noise, later ones signal imminent failure. Together they outperform either alone on this dataset.

**1D-CNN baseline**
Three Conv1d blocks with ReLU, global average pooling, then the same regression head. Faster to train (~6 min vs 18 min on T4) and converges to similar RMSE, which makes sense for FD001 — it's single operating condition, so the patterns are consistent enough for a CNN to pick up without needing long memory.

**Health Index**
Rather than feeding raw sensor values directly, I compute a composite scalar Health Index per cycle. It's a weighted average of the 14 sensors where each weight is the absolute Pearson correlation between that sensor and RUL. Sensors that track degradation closely get more weight. This follows the NASA PHM framework HI construction methodology and gives the model a cleaner signal to work with alongside the raw channels.

---

## Results

| Model | RMSE (cycles) | MAE (cycles) |
|---|---|---|
| LSTM + Attention | **13.95** | **10.73** |
| 1D-CNN | 14.82 | 10.67 |
| Literature baseline (LSTM) | 12–18 | — |

RMSE of 13.95 on CMAPSS FD001 is within the competitive range reported in published work. The predicted vs actual plot shows a clean diagonal with slight under-prediction at high RUL values (engines with lots of life left), which is expected — the model is conservative, which is the safer failure mode for maintenance decisions.

![Evaluation Results](assets/eval_results.png)

---

## What the plots show

**Predicted vs Actual (left):** Each dot is one of the 100 test engines. Points on the red dashed line are perfect predictions. The model does well at low RUL (0–40 cycles) which is the safety-critical region — you want accurate predictions when an engine is close to failure, not when it has 100 cycles left.

**Error Distribution (centre):** Roughly centered at zero, slight right skew meaning the model tends to under-predict (predict less remaining life than there actually is). For maintenance scheduling, under-prediction means you pull equipment slightly early — conservative and acceptable. Over-prediction (right side) is the dangerous failure mode.

**Training Curves (right):** Both models converge by ~epoch 75. CNN converges faster initially, LSTM catches up and edges ahead by epoch 200. The gap closes because FD001 has a single operating condition — on FD002/FD003/FD004 (multiple conditions) the LSTM advantage would be larger.

---

## How to run

The full pipeline runs in a single Colab notebook. No local setup needed.

**1.** Open `rul_predictor_full.ipynb` in Google Colab

**2.** Set runtime to T4 GPU (Runtime → Change runtime type → T4)

**3.** Run cells top to bottom. Cell 3 downloads the NASA dataset automatically. No manual file uploads needed.

**4.** Expected training time: ~18 min (LSTM) + ~7 min (CNN) on T4

**Expected output after Cell 10:**
```
=== FINAL RESULTS ===
LSTM  — RMSE: 13.95 | MAE: 10.73
CNN1D — RMSE: 14.82 | MAE: 10.67
Winner: LSTM
```

---

## Stack

| Component | Tech |
|---|---|
| Models | PyTorch (LSTM, TransformerEncoder, Conv1d) |
| Data pipeline | Pandas, NumPy, scikit-learn (MinMaxScaler) |
| Experiment tracking | Weights & Biases (wandb) |
| Visualisation | Matplotlib |
| Runtime | Google Colab, T4 GPU |

---

## What I'd add next / how to take this further

**Better uncertainty quantification.** Right now the model gives a point estimate — one number. Real PHM systems need confidence bounds. Monte Carlo Dropout (keep dropout active at inference, run 50 forward passes, report mean ± std) is a one-line change that adds calibrated uncertainty intervals. An aerospace system legally can't act on a point estimate alone.

**Multi-condition generalisation.** FD001 is the easy subset — single operating condition, single fault mode. FD002 and FD004 have 6 operating conditions. Running on those and showing the RMSE gap between LSTM and CNN widen would be a stronger result to report.

**Deployment as an inference API.** The model is 128-dim hidden state — lightweight enough to serve as a REST endpoint. FastAPI + uvicorn, serialize the model with `torch.jit.trace`, wrap it in a `/predict` endpoint that takes a 30×15 sequence and returns predicted RUL. Add a WebSocket endpoint and you can stream live predictions as new sensor data arrives — same architecture I used in the RUL dashboard project.

**NASA CMAPSS → real sensor data.** The pipeline is dataset-agnostic. Swap the data loader for any multi-channel time-series (EV battery discharge curves, CNC vibration sensors, bearing temperature logs) and the rest of the code runs unchanged. The RUL labeling and Health Index construction are the only parts that need domain-specific tuning.

---

## Files

```
rul-predictor/
├── rul_predictor_full.ipynb   # complete pipeline, run top to bottom
├── assets/
│   └── eval_results.png       # predicted vs actual + error dist + training curves
└── README.md
```

Trained checkpoints (`lstm_rul.pth`, `cnn_rul.pth`) are not included due to file size. Re-run Cell 8–9 to reproduce them in ~25 minutes on a free T4.
