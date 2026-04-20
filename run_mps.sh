#!/usr/bin/env bash
# Entropy-Driven Self-Distillation — Apple Silicon MPS
# Target: val_loss ≤ 3.3821  |  Device: MPS (M-series, 32GB)
#
# Tokens per step = batch_size × sequence_length × grad_accumulation_steps
#                 = 16 × 1024 × 32 = 524,288  (same as baseline)
# Total tokens    = 524,288 × 4768 ≈ 2.5B

python train_gpt2.py \
  --input_bin     "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin"  \
  --output_dir    pylog124M_mps \
  --model         d12 \
  --batch_size    16  \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --num_iterations  4768 \
  --learning_rate   0.0018 \
  --warmup_iters    256  \
  --warmdown_iters  1024 \
  --weight_decay    0.1  \
  --val_loss_every  128  \
  --val_batch_size  16   \
  --distill \
  --distill_keep_ratio        0.70 \
  --distill_min_keep          0.30 \
  --distill_snapshot_interval 64   \
  --distill_buffer_size       8    \
  --distill_collapse_threshold 1.25
