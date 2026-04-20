#!/usr/bin/env bash
# Entropy-Driven Self-Distillation — Apple Silicon MPS
# Target: val_loss ≤ 3.3821  |  Device: MPS (M5, 32GB)
#
# Tokens per step = batch_size × sequence_length × grad_accumulation_steps
#                 = 8 × 1024 × 64 = 524,288  (same as RTX 4090 baseline)
# Total tokens    = 524,288 × 4768 ≈ 2.5B
#
# Changes vs crashed run:
#   - MTP disabled (was --mtp 2 → 3.07 GiB extra cross_entropy = OOM)
#   - Float32 on MPS (float16 autocast → CPU fallbacks → 3378s/step)
#   - batch_size 8 (was 16) + grad_accum 64 (was 32) = same token throughput
#   - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to allow full unified memory use

export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python -u train_gpt2.py \
  --input_bin     "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin"  \
  --output_dir    pylog124M_mps \
  --model         d12 \
  --batch_size    8   \
  --grad_accumulation_steps 64 \
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
