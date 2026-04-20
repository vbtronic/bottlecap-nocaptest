import os
import sys
import uuid
import math
import glob
import time
import queue
import threading
import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

with open(sys.argv[0]) as f:
    code = f.read()

# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_synchronize(device_type: str):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def make_autocast_ctx(device_type: str):
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# =============================================================================
# MODEL ARCHITECTURE  (unchanged from baseline)
# =============================================================================

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
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


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
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head:  int = 12
    n_embd:  int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=betas,
        )


# =============================================================================
# DATA LOADING  (device-agnostic — returns CPU tensors)
# =============================================================================

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in data .bin file!"); exit(1)
    assert header[1] == 1, "unsupported version"
    return header[2]


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank  = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"no files matching {filename_pattern}"
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: {ntok_total:,} tokens across {len(self.files)} files")
        self.reset()

    def reset(self):
        self.current_shard    = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard    = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y   # CPU tensors; caller moves to device


# =============================================================================
# ENTROPY-DRIVEN SELF-DISTILLATION
# =============================================================================

class DistillationController:
    """
    Tracks the train/val loss divergence and adapts distillation aggressiveness.

    Rules:
      val/train > collapse_threshold  →  relax (increase keep_ratio)
      val/train < recovery_threshold  →  tighten (decrease keep_ratio)
    """

    def __init__(
        self,
        initial_keep_ratio: float = 0.70,
        min_keep_ratio:     float = 0.30,
        max_keep_ratio:     float = 1.00,
        collapse_threshold: float = 1.25,
        recovery_threshold: float = 1.05,
        ema_alpha:          float = 0.95,
    ):
        self.keep_ratio         = initial_keep_ratio
        self.min_keep_ratio     = min_keep_ratio
        self.max_keep_ratio     = max_keep_ratio
        self.collapse_threshold = collapse_threshold
        self.recovery_threshold = recovery_threshold
        self.ema_alpha          = ema_alpha
        self._ema_train: float | None = None
        self._ema_val:   float | None = None

    def observe_train(self, loss: float):
        if self._ema_train is None:
            self._ema_train = loss
        else:
            a = self.ema_alpha
            self._ema_train = a * self._ema_train + (1 - a) * loss

    def observe_val(self, loss: float):
        self._ema_val = loss

    def step(self) -> float:
        """Adjust keep_ratio and return new value."""
        if self._ema_train is None or self._ema_val is None:
            return self.keep_ratio
        ratio = self._ema_val / (self._ema_train + 1e-8)
        if ratio > self.collapse_threshold:
            self.keep_ratio = min(self.max_keep_ratio, self.keep_ratio + 0.05)
            print0(f"  [distill] collapse risk (val/train={ratio:.3f}) → keep_ratio={self.keep_ratio:.2f}")
        elif ratio < self.recovery_threshold:
            self.keep_ratio = max(self.min_keep_ratio, self.keep_ratio - 0.01)
        return self.keep_ratio


class DistillationDataLoader:
    """
    Asynchronous self-distillation pipeline.

    A background thread continuously pre-scores raw batches from the base
    loader using a periodic *CPU* copy of the training model.  For each
    (B, T) batch it computes per-token Shannon entropy; targets of the
    bottom (1 - keep_ratio) fraction are masked to -1 (ignored by
    cross_entropy).  The main thread pulls already-distilled batches from
    a queue and moves them to the training device, keeping the GPU/MPS
    fully saturated.

    Entropy = uncertainty = information value.
    High-entropy tokens  → model doesn't know them → keep, learn.
    Low-entropy  tokens  → model finds them trivial → mask, skip.
    """

    def __init__(
        self,
        base_loader:       DistributedDataLoader,
        model_config:      GPTConfig,
        controller:        DistillationController,
        buffer_size:       int = 8,
        snapshot_interval: int = 64,
        train_device:      str = "cpu",
    ):
        self.base_loader       = base_loader
        self.controller        = controller
        self.snapshot_interval = snapshot_interval
        self.train_device      = train_device

        # CPU snapshot model (float32, eval mode, never receives gradients)
        self._cpu_model = GPT(model_config)
        self._cpu_model.eval()
        self._snap_lock         = threading.Lock()
        self._snap_ready        = False   # True once first snapshot is loaded

        # Running stats (written only by worker thread → no lock needed)
        self.avg_keep_ratio        = 1.0
        self.avg_entropy_threshold = 0.0

        # Async queue
        self._q    = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="distill-worker"
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Called from the MAIN thread                                          #
    # ------------------------------------------------------------------ #

    def update_snapshot(self, model: nn.Module, step: int):
        """Copy training-model weights to the CPU snapshot (every N steps)."""
        if step % self.snapshot_interval != 0:
            return
        raw = model.module if hasattr(model, "module") else model
        cpu_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
        with self._snap_lock:
            self._cpu_model.load_state_dict(cpu_state)
            self._snap_ready = True

    def next_batch(self):
        """Pull next distilled batch and move to training device."""
        x, y = self._q.get()
        return x.to(self.train_device), y.to(self.train_device)

    def reset(self):
        self.base_loader.reset()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    # BACKGROUND WORKER                                                    #
    # ------------------------------------------------------------------ #

    def _worker(self):
        while not self._stop.is_set():
            try:
                x, y = self.base_loader.next_batch()   # CPU tensors
                x_d, y_d = self._distill(x, y)
                self._q.put((x_d, y_d))                # blocks if queue full
            except Exception:
                import traceback; traceback.print_exc()
                try:
                    self._q.put((x, y))
                except Exception:
                    pass

    def _distill(self, x: torch.Tensor, y: torch.Tensor):
        keep_ratio = self.controller.keep_ratio
        if keep_ratio >= 1.0 or not self._snap_ready:
            return x, y

        # Forward pass on CPU (non-blocking for the GPU/MPS)
        with self._snap_lock:
            with torch.no_grad():
                logits, _ = self._cpu_model(x, return_logits=True)

        # Per-token Shannon entropy  H(t) = -Σ p·log p
        probs   = torch.softmax(logits.float(), dim=-1)        # (B, T, V)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(-1)  # (B, T)

        # Keep the top `keep_ratio` tokens ranked by entropy
        flat    = entropy.reshape(-1)
        n_total = flat.numel()
        n_keep  = max(1, int(keep_ratio * n_total))
        k_rank  = n_total - n_keep                    # (n_total - n_keep)-th smallest
        if k_rank <= 0:
            threshold = flat.min().item() - 1.0
        else:
            threshold = torch.kthvalue(flat, k_rank).values.item()

        mask   = entropy > threshold                  # True = informative = keep
        actual = mask.float().mean().item()

        # EMA stats (only worker thread writes these)
        self.avg_keep_ratio        = 0.95 * self.avg_keep_ratio        + 0.05 * actual
        self.avg_entropy_threshold = 0.95 * self.avg_entropy_threshold + 0.05 * threshold

        y_d = y.clone()
        y_d[~mask] = -1   # cross_entropy ignore_index
        return x, y_d


# =============================================================================
# UTILITIES
# =============================================================================

VAL_TOKENS = 1_048_576


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    print0(f"Running pytorch {torch.__version__}")

    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input_bin",     type=str, default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--input_val_bin", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--output_dir",    type=str, default="")
    parser.add_argument("--model",         type=str, default="d12",
                        choices=["d12", "d24", "d36", "d48"])
    # Token layout
    parser.add_argument("--batch_size",             type=int,   default=4)
    parser.add_argument("--grad_accumulation_steps",type=int,   default=1)
    parser.add_argument("--sequence_length",        type=int,   default=64)
    # Training
    parser.add_argument("--num_iterations", type=int,   default=10)
    parser.add_argument("--learning_rate",  type=float, default=1e-4)
    parser.add_argument("--warmup_iters",   type=int,   default=0)
    parser.add_argument("--warmdown_iters", type=int,   default=0)
    parser.add_argument("--weight_decay",   type=float, default=0.0)
    # Evaluation
    parser.add_argument("--val_loss_every", type=int,   default=0)
    parser.add_argument("--val_batch_size", type=int,   default=16)
    parser.add_argument("--save_every",     type=int,   default=5000)
    # Distillation
    parser.add_argument("--distill",                  action="store_true",
                        help="Enable entropy-driven self-distillation")
    parser.add_argument("--distill_keep_ratio",       type=float, default=0.70,
                        help="Fraction of tokens to keep [0.3, 1.0]")
    parser.add_argument("--distill_min_keep",         type=float, default=0.30)
    parser.add_argument("--distill_buffer_size",      type=int,   default=8)
    parser.add_argument("--distill_snapshot_interval",type=int,   default=64,
                        help="Update CPU snapshot every N steps")
    parser.add_argument("--distill_collapse_threshold",type=float,default=1.25,
                        help="val/train ratio above which distillation is relaxed")
    # Misc
    parser.add_argument("--compile",   action="store_true", help="torch.compile (CUDA only)")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--device",    type=str, default="",
                        help="Force device (cuda/mps/cpu). Auto-detected if empty.")
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}

    # ── Device ────────────────────────────────────────────────────────────
    device      = args.device.lower() if args.device else detect_device()
    device_type = device.split(":")[0]   # 'cuda' / 'mps' / 'cpu'
    print0(f"device: {device}")

    # ── Distributed (CUDA + torchrun only) ────────────────────────────────
    use_ddp = device_type == "cuda" and "RANK" in os.environ
    if use_ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed import init_process_group, destroy_process_group
        init_process_group(backend="nccl")
        ddp_rank       = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        assert args.grad_accumulation_steps % ddp_world_size == 0
        args.grad_accumulation_steps //= ddp_world_size
        device      = f"cuda:{ddp_local_rank}"
        device_type = "cuda"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

    torch.manual_seed(42)

    # ── wandb ────────────────────────────────────────────────────────────
    if args.log_wandb and master_process:
        import wandb, datetime
        wandb.init(
            project="benchmark_gpt2",
            name=f"edsd-{args.model}-{datetime.datetime.now():%m%d-%H%M}",
        )
        wandb.config.update(args)
        wandb.save("train_gpt2.py")
        wandb.save("run.sh")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # ── Autocast + inductor ───────────────────────────────────────────────
    ctx = make_autocast_ctx(device_type)
    if device_type == "cuda":
        try:
            import torch._inductor.config as ic
            if hasattr(ic, "coordinate_descent_tuning"):
                ic.coordinate_descent_tuning = True
        except ImportError:
            pass

    # ── Data loaders ──────────────────────────────────────────────────────
    train_loader_raw = DistributedDataLoader(
        args.input_bin, B, T, ddp_rank, ddp_world_size
    )
    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0, \
        f"VAL_TOKENS={VAL_TOKENS} not divisible by {tokens_per_iter_val}"
    val_steps = VAL_TOKENS // tokens_per_iter_val

    # ── Model ─────────────────────────────────────────────────────────────
    model_configs = {
        "d12": GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }
    model_config = model_configs[args.model]
    model = GPT(model_config).train().to(device)

    if args.compile:
        if device_type == "cuda":
            print0("compiling the model...")
            model = torch.compile(model)
        else:
            print0("--compile ignored on non-CUDA device")

    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if use_ddp else model

    # ── Distillation ──────────────────────────────────────────────────────
    controller   = None
    train_loader = train_loader_raw

    if args.distill:
        controller = DistillationController(
            initial_keep_ratio    = args.distill_keep_ratio,
            min_keep_ratio        = args.distill_min_keep,
            collapse_threshold    = args.distill_collapse_threshold,
        )
        train_loader = DistillationDataLoader(
            base_loader        = train_loader_raw,
            model_config       = model_config,
            controller         = controller,
            buffer_size        = args.distill_buffer_size,
            snapshot_interval  = args.distill_snapshot_interval,
            train_device       = device,
        )
        print0(
            f"[distill] ENABLED  keep_ratio={args.distill_keep_ratio:.2f}  "
            f"snapshot_every={args.distill_snapshot_interval}  "
            f"buffer={args.distill_buffer_size}"
        )
    else:
        print0("[distill] DISABLED — baseline mode")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = raw_model.configure_optimizers(
        weight_decay  = args.weight_decay,
        learning_rate = args.learning_rate,
        betas         = (0.9, 0.95),
        device_type   = device_type,
    )

    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        else:
            return args.learning_rate * (args.num_iterations - it) / args.warmdown_iters

    # ── Logging ───────────────────────────────────────────────────────────
    run_id  = str(uuid.uuid4())
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, f"{run_id}.log")
        open(logfile, "w").close()

    # ── Training loop ─────────────────────────────────────────────────────
    training_time_ms = 0.0
    device_synchronize(device_type)
    t0 = time.perf_counter()

    # Pre-fetch first batch
    next_x, next_y = train_loader.next_batch()

    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        # ── Validation (always on RAW data) ───────────────────────────────
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            device_synchronize(device_type)
            training_time_ms += 1000 * (time.perf_counter() - t0)

            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = torch.zeros(1, device=device)
                for _ in range(val_steps):
                    xv, yv = val_loader.next_batch()
                    xv, yv = xv.to(device), yv.to(device)
                    _, loss = model(xv, yv, return_logits=False)
                    val_loss += loss
                if use_ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss_scalar = (val_loss / val_steps).item()

            print0(f"step:{step}/{args.num_iterations} | val_loss {val_loss_scalar:.6f}")

            # Distillation anti-collapse adjustment
            if controller is not None:
                controller.observe_val(val_loss_scalar)
                new_ratio = controller.step()
                if isinstance(train_loader, DistillationDataLoader):
                    print0(
                        f"  [distill] keep_ratio={new_ratio:.3f}  "
                        f"avg_kept={train_loader.avg_keep_ratio:.3f}  "
                        f"H_thr={train_loader.avg_entropy_threshold:.3f}"
                    )

            if master_process:
                if args.log_wandb:
                    log_dict = {
                        "val_loss":  val_loss_scalar,
                        "time_ms":   training_time_ms,
                    }
                    if controller is not None:
                        log_dict["keep_ratio"] = controller.keep_ratio
                    wandb.log(log_dict, step=step * tokens_per_iter)
                if logfile:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} val:{val_loss_scalar:.6f}\n")

            device_synchronize(device_type)
            t0 = time.perf_counter()

        if last_step:
            break

        # ── Update CPU snapshot for distillation worker ───────────────────
        if isinstance(train_loader, DistillationDataLoader):
            train_loader.update_snapshot(model, step)

        # ── Training step ─────────────────────────────────────────────────
        model.train()
        train_loss = torch.zeros(1, device=device)

        for micro_step in range(args.grad_accumulation_steps):
            if use_ddp:
                model.require_backward_grad_sync = (
                    micro_step == args.grad_accumulation_steps - 1
                )
            x = next_x.to(device)
            y = next_y.to(device)
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / args.grad_accumulation_steps
                train_loss += loss.detach()
            # Pre-fetch next while GPU computes backward
            next_x, next_y = train_loader.next_batch()
            loss.backward()

        # Optimizer
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Diagnostics ───────────────────────────────────────────────────
        device_synchronize(device_type)
        approx_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        if use_ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()

        if controller is not None:
            controller.observe_train(lossf)

        distill_suffix = ""
        if isinstance(train_loader, DistillationDataLoader):
            distill_suffix = f" | kept:{train_loader.avg_keep_ratio:.2f}"

        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | "
            f"lr:{lr:.2e} | train_time:{approx_ms/1000:.2f}s | "
            f"step_avg:{approx_ms/(step+1):.2f}ms"
            + distill_suffix
        )

        if master_process and logfile:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trn:{lossf:.6f}\n")

        if master_process and (step + 1) % args.save_every == 0:
            ckpt = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(ckpt, f"logs/{run_id}/model_step{step:06d}.pt")

    # ── Peak memory report ────────────────────────────────────────────────
    if device_type == "cuda":
        print0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    elif device_type == "mps":
        try:
            print0(f"peak memory: {torch.mps.driver_allocated_memory() // 1024 // 1024} MiB")
        except Exception:
            pass

    # ── Save final checkpoint ─────────────────────────────────────────────
    if master_process:
        ckpt = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch.save(ckpt, f"logs/{run_id}/final.pt")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if isinstance(train_loader, DistillationDataLoader):
        train_loader.stop()
    if use_ddp:
        destroy_process_group()
