#!/usr/bin/env bash
# The upstream baseline (same command as BottleCapAI/NoCap-Test run.sh) plus early stopping.
torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --target_val_loss 3.3821 \
  "$@"
