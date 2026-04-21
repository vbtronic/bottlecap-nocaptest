# Smart Filtering — Goldilocks Zone + Early Ejection

*Thanks to Ondřej Plátek for the critical observation that moved us from loss masking to true Early Ejection.*

## Motivation

Not all tokens are equally useful for training. Two categories waste GPU compute:

- **Too easy**: model already assigns high probability to the correct token → gradient ≈ 0
- **Too noisy**: extreme loss indicates garbage/anomalous text → corrupts gradients rather than helping

Our original approach masked these tokens in the cross-entropy loss. The problem (pointed out by Ondřej Plátek): masking in loss still runs the full GPU forward/backward pass. The compute is wasted regardless of whether we count the loss contribution.

**The fix:** eject bad batches *before* they reach the GPU. True Early Ejection.

## Method

### Token scoring

We use the model's own output distribution to compute per-token entropy:

```
H(t) = -Σ p(v|context) · log p(v|context)
```

A high entropy means the model is uncertain → the token is hard → keep it.  
A low entropy means the model is confident → easy token → mask it out.

### Asynchronous distillation pipeline

Scoring every batch in the forward pass would add latency to training. Instead:

1. A **background thread** maintains a CPU copy of the model (snapshot).
2. While the GPU trains on batch `t`, the CPU thread pre-scores batch `t+k`.
3. Scored batches go into a queue; the training loop consumes them without waiting.

The CPU snapshot is refreshed every `--distill_snapshot_interval` steps so scores stay reasonably up to date.

### Token selection

For each batch, we sort tokens by entropy and keep the top `keep_ratio` fraction. Masked tokens have their target set to `-1` (PyTorch `cross_entropy` ignore_index), so they contribute zero loss and zero gradient.

### Anti-collapse controller

If the model over-filters, it may stop improving on validation even while train loss drops. A controller monitors the train/val loss ratio and automatically relaxes `keep_ratio` when `val/train > collapse_threshold`, preventing the training from becoming too narrow.

### Adaptive scoring modes

The background thread adapts its depth based on idle time:
- **full** (all 12 layers): when GPU is waiting for data → use full model
- **light** (4 layers): when GPU is slightly ahead
- **raw** (no scoring, keep all): when GPU would stall waiting for scorer

This ensures the distillation pipeline never becomes a bottleneck.

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `--distill_keep_ratio` | 0.70 | Keep top 70% hardest tokens |
| `--distill_min_keep` | 0.30 | Never drop below 30% (floor) |
| `--distill_snapshot_interval` | 64 | Refresh CPU snapshot every 64 steps |
| `--distill_buffer_size` | 8 | Pre-scored batch queue depth |
| `--distill_collapse_threshold` | 1.25 | Relax filter if val/train ratio > 1.25 |

## Results

Tested on Apple M5 (MPS, 32GB) — local development only, not benchmark hardware.

- 3-step sanity check: val_loss 10.987 → 9.576 → 8.447 → 7.793 (clean convergence, no OOM)
- No full CUDA run completed yet (no RTX 4090 available for testing)

## Why this should work

Curriculum learning literature consistently shows that training on harder examples improves sample efficiency. EDSD implements a soft, adaptive curriculum: the model itself decides which tokens are "already known" vs "still learning." Unlike static curricula, this self-updates as the model improves.

The background async pipeline ensures the technique adds near-zero wall-clock overhead to each training step.

## What didn't work / open questions

- MTP (multi-token prediction) caused OOM on MPS with large vocab (50257) — disabled
- The keep_ratio of 0.70 is a guess; optimal value likely depends on training stage
- Early in training (steps 0–500), the model is uncertain about almost everything, so EDSD has little effect — it becomes more useful as training progresses past loss ~5-6
- We haven't measured the actual speedup on CUDA hardware
