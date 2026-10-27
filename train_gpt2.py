import os
import sys
import uuid
import math
import glob
import time
import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

with open(sys.argv[0]) as f:
    code = f.read()

sys.stdout.reconfigure(line_buffering=True)  # show progress live when piped (e.g. into tee on Colab)

# -----------------------------------------------------------------------------
# Device helpers (CUDA for real runs, MPS/CPU only for smoke tests)


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_synchronize(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def pick_amp_dtype(device_type):
    """bf16 on Ampere+ (A100/L4/RTX 30xx+), fp16 on older GPUs (T4/V100), none elsewhere."""
    if device_type != "cuda":
        return None
    # is_bf16_supported() also reports emulated bf16 (True on a T4), so check the architecture instead
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model (unchanged from the baseline)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None, return_logits=True, per_sequence=False):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            if per_sequence:
                # (B,) mean loss of every sequence; used by the smart filter for scoring
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).view(targets.shape).mean(dim=1)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None
        return logits, loss


# -----------------------------------------------------------------------------
# Data loader — returns CPU tensors, the caller moves them to the device


def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return header[2]


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match {filename_pattern}"
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        self.tokens_served = 0  # tokens handed out by this rank (incl. ones the filter drops)
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
        self.reset()

    # current_position is shared by all ranks; rank r reads the r-th slice of each chunk,
    # so the batch size may change between calls (the smart filter reads bigger pools)
    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, B=None):
        """Returns B sequences (default self.B) as CPU tensors."""
        B = B or self.B
        T = self.T
        chunk = B * T * self.num_processes
        if self.current_position + chunk + 1 > len(self.tokens):
            self.advance()
        start = self.current_position + self.process_rank * B * T
        buf = torch.from_numpy(self.tokens[start : start + B * T + 1].astype(np.int64))
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += chunk
        self.tokens_served += B * T
        return x, y

    def state_dict(self):
        return dict(shard=self.current_shard, position=self.current_position,
                    tokens_served=self.tokens_served)

    def load_state_dict(self, state):
        self.current_shard = state["shard"]
        self.current_position = state["position"]
        self.tokens_served = state["tokens_served"]
        self.tokens = _load_data_shard(self.files[self.current_shard])


# -----------------------------------------------------------------------------
# Smart Filtering — Goldilocks zone + Early Ejection
#
# For every micro-batch we draw a candidate pool of ceil(B * pool_factor) sequences,
# score each one with the *live* model (no grad, only the first score_len tokens, so
# scoring costs a small fraction of a training step), and keep the B sequences in the
# middle of the loss ranking. The easiest and the hardest candidates are ejected before
# the training forward/backward, so the GPU never spends a backward pass on them.
#
#   too easy  (lowest loss)  — the model already predicts them, ~no gradient signal
#   too hard  (highest loss) — mostly noise: tables, code dumps, other languages, spam


class SmartFilter:
    def __init__(self, score_model, B, pool_factor, hard_frac, score_len, ctx):
        self.score_model = score_model
        self.B = B
        self.pool = max(B, math.ceil(B * pool_factor))
        n_drop = self.pool - B
        self.n_hard = round(n_drop * hard_frac)
        self.n_easy = n_drop - self.n_hard
        self.score_len = score_len
        self.ctx = ctx
        # running stats for logging
        self.loss_kept = 0.0
        self.loss_easy = 0.0
        self.loss_hard = 0.0
        self.n_batches = 0

    @torch.no_grad()
    def select(self, x, y):
        """x, y: (pool, T) on device -> (B, T) slices of the kept sequences."""
        with self.ctx:
            _, scores = self.score_model(
                x[:, : self.score_len], y[:, : self.score_len],
                return_logits=False, per_sequence=True,
            )
        order = torch.argsort(scores)
        keep = order[self.n_easy : self.n_easy + self.B]
        self.n_batches += 1
        a = 1.0 / min(self.n_batches, 100)  # running mean, then EMA
        sorted_scores = scores[order]
        self.loss_kept += a * (sorted_scores[self.n_easy : self.n_easy + self.B].mean().item() - self.loss_kept)
        if self.n_easy:
            self.loss_easy += a * (sorted_scores[: self.n_easy].mean().item() - self.loss_easy)
        if self.n_hard:
            self.loss_hard += a * (sorted_scores[self.pool - self.n_hard :].mean().item() - self.loss_hard)
        return x[keep], y[keep]


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. Keep fixed for consistent comparisons


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    print0(f"Running pytorch {torch.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--input_val_bin", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--output_dir", type=str, default="", help="where to write the .log file")
    parser.add_argument("--ckpt_dir", type=str, default="logs", help="where to write checkpoints")
    parser.add_argument("--model", type=str, default="d12", choices=["d12", "d24", "d36", "d48"])
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=64)
    # workload
    parser.add_argument("--num_iterations", type=int, default=10)
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--warmdown_iters", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="0 = off (baseline)")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--target_val_loss", type=float, default=0.0,
                        help="stop as soon as val_loss <= this (0 = off)")
    # checkpointing
    parser.add_argument("--save_every", type=int, default=0, help="0 = only final checkpoint")
    parser.add_argument("--resume", type=str, default="", help="checkpoint .pt to resume from")
    # smart filtering
    parser.add_argument("--smart_filter", action="store_true", help="enable Goldilocks filtering")
    parser.add_argument("--filter_pool", type=float, default=1.25,
                        help="candidates scored per kept sequence (1.25 = eject 20%%)")
    parser.add_argument("--filter_hard_frac", type=float, default=0.5,
                        help="fraction of ejected sequences taken from the hard end")
    parser.add_argument("--filter_score_len", type=int, default=256,
                        help="score only the first N tokens of each candidate")
    parser.add_argument("--filter_warmup", type=int, default=128,
                        help="no filtering for the first N steps (model scores are noise)")
    # misc
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--log_wandb", action="store_true")
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length
    assert args.filter_pool >= 1.0 and 0.0 <= args.filter_hard_frac <= 1.0
    assert 0 < args.filter_score_len <= T

    # ── Device / DDP ──────────────────────────────────────────────────────
    use_ddp = "RANK" in os.environ
    if use_ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        assert torch.cuda.is_available(), "DDP needs CUDA"
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        assert args.grad_accumulation_steps % ddp_world_size == 0
        args.grad_accumulation_steps //= ddp_world_size
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        device = args.device or detect_device()
    device_type = device.split(":")[0]
    master_process = ddp_rank == 0
    print0(f"using device: {device}" + (f" ({torch.cuda.get_device_name()})" if device_type == "cuda" else ""))

    torch.manual_seed(42)

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    amp_dtype = pick_amp_dtype(device_type)
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    print0(f"autocast dtype: {amp_dtype}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size)
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val

    # ── Model ─────────────────────────────────────────────────────────────
    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768),  # 124M GPT-2
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    raw_model = GPT(model_config).train().to(device)
    model = raw_model
    if device_type == "cuda" and not args.no_compile:
        import torch._inductor.config as inductor_config

        if hasattr(inductor_config, "coordinate_descent_tuning"):
            inductor_config.coordinate_descent_tuning = True
        print0("compiling the model...")
        # dynamic=False: training (B,T) and scoring (pool, score_len) get separate static graphs
        model = torch.compile(model, dynamic=False)
    compiled_model = model  # used for scoring: bypasses DDP hooks
    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        else:
            return args.learning_rate * (args.num_iterations - it) / args.warmdown_iters

    smart_filter = None
    if args.smart_filter:
        smart_filter = SmartFilter(
            compiled_model, B, args.filter_pool, args.filter_hard_frac, args.filter_score_len, ctx
        )
        print0(f"[filter] pool={smart_filter.pool} keep={B} eject easy={smart_filter.n_easy} "
               f"hard={smart_filter.n_hard} score_len={args.filter_score_len} warmup={args.filter_warmup}")
    else:
        print0("[filter] off — baseline")

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    training_time_ms = 0.0
    run_id = str(uuid.uuid4())
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = ckpt["step"] + 1
        training_time_ms = ckpt["training_time_ms"]
        run_id = ckpt["run_id"]
        print0(f"resumed {args.resume} at step {start_step}, train_time so far {training_time_ms / 1000:.1f}s")

    if args.log_wandb and master_process:
        import wandb
        import datetime

        name = "smartfilter" if args.smart_filter else "baseline"
        wandb.init(project="benchmark_gpt2", name=f"{name}-{args.model} {datetime.datetime.now():%Y-%m-%d %H:%M}",
                   id=run_id, resume="allow")
        wandb.config.update(vars(args), allow_val_change=True)

    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, f"{run_id}.log")
        if not args.resume:
            open(logfile, "w").close()

    def save_checkpoint(step, name):
        ckpt_path = os.path.join(args.ckpt_dir, run_id)
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(dict(
            model=raw_model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            train_loader=train_loader.state_dict(), step=step, training_time_ms=training_time_ms_now(),
            run_id=run_id, args=vars(args), code=code,
        ), os.path.join(ckpt_path, name))
        print0(f"checkpoint -> {os.path.join(ckpt_path, name)}")

    def fetch_batch(step):
        """Next micro-batch on device. With the filter on, this includes scoring (timed as training)."""
        filtering = smart_filter is not None and step >= args.filter_warmup
        x, y = train_loader.next_batch(smart_filter.pool if filtering else B)
        x = x.pin_memory().to(device, non_blocking=True) if device_type == "cuda" else x.to(device)
        y = y.pin_memory().to(device, non_blocking=True) if device_type == "cuda" else y.to(device)
        if filtering:
            x, y = smart_filter.select(x, y)
        return x, y

    # ── Training loop ─────────────────────────────────────────────────────
    device_synchronize(device_type)
    t0 = time.perf_counter()

    def training_time_ms_now():
        return training_time_ms + 1000 * (time.perf_counter() - t0)

    for step in range(start_step, args.num_iterations + 1):
        last_step = step == args.num_iterations

        # once in a while evaluate on the (always unfiltered) validation set
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            device_synchronize(device_type)
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = torch.zeros((), device=device)
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val.to(device), y_val.to(device), return_logits=False)
                    val_loss += loss
                if use_ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss = (val_loss / val_steps).item()
            tokens_seen = train_loader.tokens_served * ddp_world_size
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f} | "
                   f"train_time:{training_time_ms / 1000:.1f}s | tokens_read:{tokens_seen / 1e6:.1f}M")
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss, "time": training_time_ms, "tokens_read": tokens_seen},
                              step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} val:{val_loss:f} time:{training_time_ms / 1000:.1f}\n")
            if args.target_val_loss > 0 and val_loss <= args.target_val_loss:
                print0(f"target val loss {args.target_val_loss} reached at step {step} "
                       f"after {training_time_ms / 1000 / 3600:.3f}h of training")
                device_synchronize(device_type)
                t0 = time.perf_counter()
                break
            device_synchronize(device_type)
            t0 = time.perf_counter()

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros((), device=device)
        x, y = fetch_batch(step)
        for micro_step in range(args.grad_accumulation_steps):
            sync = micro_step == args.grad_accumulation_steps - 1
            with model.no_sync() if use_ddp and not sync else contextlib.nullcontext():
                with ctx:
                    _, loss = model(x, y, return_logits=False)
                    loss = loss / args.grad_accumulation_steps
                train_loss += loss.detach()
                if not sync:
                    x, y = fetch_batch(step)  # queue the next batch before backward
                scaler.scale(loss).backward()

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------

        device_synchronize(device_type)
        approx_training_time_ms = training_time_ms_now()
        if use_ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        filter_tag = ""
        if smart_filter is not None and step >= args.filter_warmup:
            sf = smart_filter
            filter_tag = f" | score kept:{sf.loss_kept:.2f} easy:{sf.loss_easy:.2f} hard:{sf.loss_hard:.2f}"
        print0(f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time_ms / 1000:.2f}s"
               f" | step_avg:{approx_training_time_ms / (step + 1):.2f}ms" + filter_tag)
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trn:{lossf:f}\n")

        if master_process and args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_checkpoint(step, "latest.pt")  # overwrite: full checkpoints are ~1.5 GB each

    if device_type == "cuda":
        print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if master_process:
        save_checkpoint(step, "final.pt")

    if use_ddp:
        dist.destroy_process_group()
