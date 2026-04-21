# Entropy-Driven Self-Distillation for GPT-2

**A submission to the [BottleCapAI NoCap-Test benchmark](https://github.com/BottleCapAI/NoCap-Test).**

The core idea: not all tokens are equally useful for training. When a language model is already confident about a token, its gradient contribution is near zero. This implementation uses the model's own entropy to identify and focus training on the tokens that actually matter.

---

## The Idea

Standard cross-entropy trains uniformly on every token in every batch. Most of those tokens are "easy" — the model already assigns high probability to the correct answer and learns little from them.

**Entropy-Driven Self-Distillation (EDSD)** masks those easy tokens out:

1. A background thread runs the current model (CPU snapshot) over each upcoming batch
2. Per-token Shannon entropy is computed: `H(t) = -Σ p(v) · log p(v)`
3. Only the top `keep_ratio` most uncertain tokens are kept — the rest get `target = -1` (ignored by cross-entropy)
4. The main GPU training loop receives pre-scored batches with no added latency

The model trains on the same data, same number of steps — but every gradient update comes from tokens the model is actually still learning from.

```
batch tokens:  [easy][easy][HARD][easy][HARD][HARD][easy][HARD]
                  ↓      ↓    ↓     ↓     ↓     ↓     ↓     ↓
cross-entropy:  [ -1 ][ -1 ][loss][ -1 ][loss][loss][ -1 ][loss]
```

---

## How it works

### Async scoring pipeline

```
GPU (training)          CPU (scoring thread)
─────────────           ──────────────────────
train on batch t   ←── scored batch t (from queue)
train on batch t+1 ←── scored batch t+1
...                     scores batch t+2 using CPU snapshot
                        scores batch t+3
                        ...
```

The CPU thread never blocks the GPU. If it falls behind, the system falls back to keeping all tokens (`raw` mode).

### Adaptive scoring depth

The scorer adjusts how deeply it runs the CPU model based on how fast the GPU is consuming batches:

| GPU idle time | Scoring mode | Layers used |
|---|---|---|
| < 8ms | `full` | all 12 |
| 8–25ms | `light` | 4 |
| > 25ms | `raw` | 0 (keep all) |

### Anti-collapse controller

Over-filtering can cause training to diverge from validation. The controller monitors the train/val loss ratio and automatically relaxes the `keep_ratio` if `val/train > 1.25`, preventing overfitting to a narrow token subset.

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
| `train_gpt2.py` | Training script with EDSD implementation |
| `run.sh` | Benchmark run (CUDA) |
| `run_mps.sh` | Development run (Apple Silicon) |
| `IDEA.md` | Detailed method description |

---

*Fork of [BottleCapAI/NoCap-Test](https://github.com/BottleCapAI/NoCap-Test), which is based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).*
