# Smart Filtering for GPT-2

A submission to the [BottleCapAI NoCap-Test benchmark](https://github.com/BottleCapAI/NoCap-Test):
train GPT-2 124M on FineWeb to **val loss ≤ 3.3821** faster than the baseline (5.401 h on one RTX 4090,
4768 steps, 2.5B tokens; up to 5B tokens allowed).

**Idea:** spend the unused token budget on *reading* more data, and train only on the sequences in the
Goldilocks zone. Each micro-batch draws 25% extra candidates, the live model scores them on a short prefix,
and the easiest and noisiest ones are ejected before the forward/backward pass. Details and the history of
the first, broken version are in [IDEA.md](IDEA.md).

```
pool of 20 sequences ──► score (live model, no_grad, first 256 tokens) ──► sort by loss
                                                                             │
          eject 2 easiest ◄──────────────────────────────────────────────────┤
          eject 2 hardest ◄──────────────────────────────────────────────────┤
                                                                             ▼
                                                   train on the middle 16 (as baseline)
```

## Running

### Google Colab

Open [`colab.ipynb`](https://colab.research.google.com/github/vbtronic/bottlecap-nocaptest/blob/main/colab.ipynb),
pick a GPU runtime and run all cells. It trains the baseline and Smart Filtering on the same GPU and plots both.
The `quick` preset (157M tokens) fits a free T4; the `full` preset needs an A100/L4.

### Local CUDA GPU

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 33   # 33 × 100M tokens (Smart Filtering reads ~3.1B)
./run_baseline.sh                     # upstream baseline, same GPU, for a fair comparison
./run.sh                              # Smart Filtering
```

Both scripts stop at val loss ≤ 3.3821. Extra flags are passed through, e.g. `./run.sh --log_wandb`.
On GPUs without bf16 (T4, V100) training uses fp16 with a gradient scaler.

### Smoke test (CPU / Apple Silicon)

```bash
python train_gpt2.py --batch_size 4 --sequence_length 256 --val_batch_size 64 --val_loss_every 10 \
  --num_iterations 20 --learning_rate 0.0018 --smart_filter --filter_warmup 5 --filter_score_len 64
```

## Smart Filtering flags

| Flag | Default | Meaning |
|---|---|---|
| `--smart_filter` | off | enable filtering (off = exact baseline) |
| `--filter_pool` | 1.25 | candidates read per kept sequence |
| `--filter_hard_frac` | 0.5 | share of ejected sequences taken from the hard end |
| `--filter_score_len` | 256 | tokens per candidate used for scoring |
| `--filter_warmup` | 128 | steps without filtering at the start |

Other additions over upstream: `--target_val_loss`, `--grad_clip` (off), `--save_every` / `--resume`
(model, optimizer and data position), `--no_compile`, CPU/MPS fallback for smoke tests.

---

*Fork of [BottleCapAI/NoCap-Test](https://github.com/BottleCapAI/NoCap-Test), based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).*
