# Smart Filtering for GPT-2

**A submission to the [BottleCapAI NoCap-Test benchmark](https://github.com/BottleCapAI/NoCap-Test).**

The core idea: filter data *before* it reaches the GPU. Batches that provide no useful gradient signal are ejected at the CPU scorer — the GPU never wastes a forward/backward pass on them.

---

## The Idea

Standard training wastes GPU compute on two kinds of tokens:

- **Too easy**: the model already assigns high probability to the correct token → gradient ≈ 0
- **Too noisy**: garbage/anomalous text with extreme loss → corrupts gradients

Simply masking these tokens in the cross-entropy loss still runs the full GPU forward/backward — the compute is wasted. The real speedup comes from **Early Ejection**: deciding before the GPU ever sees a batch.

### Goldilocks Zone Filtering

We compute per-sequence entropy using a CPU snapshot of the model. Sequences outside the *Goldilocks zone* are ejected entirely:

```
Entropy distribution across batches:

 too easy │ ████ Goldilocks zone █████ │ too noisy
──────────┼──────────────────────────────┼──────────
  eject   │  ← train here only →        │  eject
          lo_threshold             hi_threshold
```

Thresholds adapt dynamically as training progresses (EMA of entropy distribution). During warmup (first 20 batches), everything is accepted.

### Two-level filtering

```
Raw batch from dataset
       │
       ▼ CPU scorer (async thread)
  ┌─────────────────────────────────────────┐
  │  1. Sequence level                      │
  │     seq_entropy = mean(token entropy)   │
  │     outside [lo, hi] → EJECT            │
  │     (GPU never sees this batch)         │
  │                                         │
  │  2. Token level (within kept sequences) │
  │     bottom keep_ratio tokens → mask -1  │
  └─────────────────────────────────────────┘
       │
       ▼ Queue → GPU
  GPU trains only on Goldilocks batches
  with residual easy tokens masked
```

### Async pipeline — GPU never waits

```
GPU (training)              CPU (scoring thread)
──────────────              ──────────────────────────────
train on batch t       ←── Goldilocks batch t
train on batch t+1     ←── Goldilocks batch t+1  (may have consumed
...                         3 raw batches to find this one)
```

The CPU thread pre-filters raw batches and only delivers Goldilocks ones. If it falls behind, scoring depth drops automatically (4 levels: deep → medium → light → ultra_light → raw).

### Adaptive scoring depth

| GPU idle time | Mode | Layers |
|---|---|---|
| ≤ 3ms | `deep` | all 12 |
| ≤ 8ms | `medium` | 6 |
| ≤ 20ms | `light` | 3 |
| ≤ 50ms | `ultra_light` | 1 |
| > 50ms | `raw` | 0 (pass-through) |

### Anti-collapse controller

If train/val diverge (over-filtering), `keep_ratio` is automatically relaxed to prevent the model from overfitting to a narrow token subset.

---

## Running

### Requirements

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py  # downloads ~100GB FineWeb dataset
```

### CUDA (benchmark — RTX 3090/4090 or similar)

```bash
./run.sh
```

This runs the full benchmark: AdamW + EDSD, 4768 steps, 2.5B tokens, stops automatically at val_loss ≤ 3.3821.

### Apple Silicon (development/testing)

```bash
./run_mps.sh
```

Adapted for MPS (Metal): float32, batch_size=8, grad_accum=64, same total token budget.

---

## Key parameters

| Flag | Default | Description |
|---|---|---|
| `--distill` | off | Enable EDSD |
| `--distill_keep_ratio` | 0.70 | Keep top 70% hardest tokens |
| `--distill_min_keep` | 0.30 | Never drop below 30% |
| `--distill_snapshot_interval` | 64 | Refresh CPU snapshot every N steps |
| `--distill_buffer_size` | 8 | Pre-scored batch queue size |
| `--distill_collapse_threshold` | 1.25 | Relax filter if val/train diverges |
| `--target_val_loss` | 0 | Stop early at this val_loss (e.g. 3.3821) |

---

## Verified behavior

3-step test on Apple M5 (MPS, 32GB):

```
step:0 | val_loss 10.987275
step:1 | val_loss  9.575774   (−1.41)
step:2 | val_loss  8.447386   (−1.13)
step:3 | val_loss  7.793463   (−0.65)
peak memory: 31.3 GiB  — no OOM
distill idle: 0.8% of walltime  — scorer keeps up
```

Loss decreases cleanly. Distillation overhead is negligible. No crashes.

Full benchmark run (4768 steps to val_loss 3.3821) requires a CUDA GPU — **results welcome**.

---

## Architecture

- GPT-2 124M (`d12`: 12 layers, 12 heads, 768 dim)
- RoPE positional embeddings
- Flash attention (`F.scaled_dot_product_attention`)
- RMSNorm (`F.rms_norm`)
- Weight tying (embedding ↔ lm_head)
- AdamW optimizer (standard baseline)
- CUDA: bfloat16 autocast | MPS: float32

---

## Files

| File | Description |
|---|---|
| `train_gpt2.py` | Training script with Smart Filtering implementation |
| `run.sh` | Benchmark run (CUDA) |
| `run_mps.sh` | Development run (Apple Silicon) |
| `IDEA.md` | Detailed method description |

---

## Acknowledgements

Thanks to **Ondřej Plátek** for the key insight that masking tokens in cross-entropy still wastes GPU compute — and that real savings require ejecting batches *before* the forward pass.

---

*Fork of [BottleCapAI/NoCap-Test](https://github.com/BottleCapAI/NoCap-Test), which is based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).*
