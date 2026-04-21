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
    # MPS: float16 autocast causes CPU fallbacks for cross_entropy/softmax → 100x slowdown.
    # Run in float32 instead; unified memory on Apple Silicon makes this affordable.
    return contextlib.nullcontext()


# =============================================================================
# MUON OPTIMIZER
# Momentum + orthogonalized update for 2-D weight matrices.
# Proven 2–3× faster convergence than AdamW in transformer pre-training.
# Reference: Keller Jordan, https://github.com/KellerJordan/modded-nanogpt
# =============================================================================

@torch.no_grad()
def _newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to orthogonalize G.
    Always operates in float32 for numerical stability.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon — Momentum + Orthogonalized Update.
    Applied to all 2-D weight matrices in the transformer.
    Embeddings and 1-D params use standard AdamW via a separate optimizer.

    Args:
        lr:       learning rate (typically 0.02, ~10× AdamW rate)
        momentum: Nesterov momentum coefficient (default 0.95)
        ns_steps: Newton-Schulz iteration steps (default 5)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr  = group["lr"]
            mom = group["momentum"]
            ns  = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = g.clone()
                buf = state["buf"]
                buf.lerp_(g, 1 - mom)                      # EMA momentum
                g_nes = g.lerp_(buf, mom)                  # Nesterov correction
                g_orth = _newtonschulz5(g_nes, steps=ns)   # orthogonalize
                # scale so update norm ≈ param norm (adaptive step)
                scale = max(1, g_nes.size(0) / g_nes.size(1)) ** 0.5
                p.data.add_(g_orth, alpha=-lr * scale)


# =============================================================================
# MODEL ARCHITECTURE
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


def rmsnorm(x, eps=1e-6):
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary  = Rotary(self.head_dim)

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
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


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
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768
    mtp_n:      int = 1     # multi-token prediction depth (1 = disabled)


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

        # Multi-token prediction heads (k=2,3,…)
        # Lightweight projectors that predict k steps ahead from same hidden state.
        if config.mtp_n > 1:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for _ in range(config.mtp_n - 1)
            ])

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
            # Multi-token prediction: predict k=2,3,… steps ahead (weight 0.5^k)
            if hasattr(self, "mtp_heads") and self.training:
                B, T = targets.shape
                for k, head in enumerate(self.mtp_heads, 2):
                    mtp_logits = head(x)
                    # Shift targets left by k, pad right with -1
                    pad = torch.full((B, k), -1, dtype=targets.dtype, device=targets.device)
                    mtp_tgt = torch.cat([targets[:, k:], pad], dim=1)
                    mtp_loss = F.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.size(-1)),
                        mtp_tgt.view(-1), ignore_index=-1,
                    )
                    loss = loss + (0.5 ** k) * mtp_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

    def make_optimizers(self, muon_lr, adamw_lr, weight_decay, betas):
        """
        Returns (muon_opt, adamw_opt).
        - Muon:  all 2-D weight matrices inside transformer blocks
        - AdamW: embeddings, lm_head, MTP heads, 1-D params
        """
        muon_params, adamw_params = [], []
        for name, p in self.named_parameters():
            # 2-D matrices inside transformer blocks → Muon
            if p.ndim == 2 and name.startswith("transformer.h."):
                muon_params.append(p)
            else:
                adamw_params.append(p)

        muon_opt  = Muon(muon_params, lr=muon_lr, momentum=0.95)
        adamw_opt = torch.optim.AdamW(
            adamw_params, lr=adamw_lr, weight_decay=weight_decay, betas=betas
        )
        return muon_opt, adamw_opt


# =============================================================================
# DATA LOADING  — returns CPU tensors; callers move to device
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
        assert header[0] == 20240520 and header[1] == 1
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
        return x, y


# =============================================================================
# SMART FILTERING — Goldilocks Zone + Early Ejection
#
# Key idea (credit: Ondřej Plátek):
#   Masking tokens in cross-entropy wastes GPU time on useless forward/backward
#   passes.  Instead, filter at the CPU scorer level so invalid batches never
#   reach the GPU at all (true Early Ejection, not just loss masking).
#
# Two-level filtering:
#   1. Sequence level  — eject entire sequences outside the Goldilocks zone
#   2. Token level     — mask residual easy tokens within kept sequences
#
# Goldilocks zone:  lo_threshold < mean_sequence_entropy < hi_threshold
#   too easy  (entropy < lo): model already knows this → no gradient signal
#   too noisy (entropy > hi): anomalous/garbage text → corrupts gradients
# =============================================================================

class GoldilocksFilter:
    """
    Tracks per-sequence entropy statistics and emits dynamic thresholds.
    Warmup phase (first `warmup` batches): accepts everything.
    """
    def __init__(self, lo_sigma=1.0, hi_sigma=2.5, ema_alpha=0.99, warmup=20):
        self.lo_sigma  = lo_sigma
        self.hi_sigma  = hi_sigma
        self.ema_alpha = ema_alpha
        self.warmup    = warmup
        self._count    = 0
        self._ema_mean = None
        self._ema_var  = None

    def update(self, seq_entropy: torch.Tensor):
        m = seq_entropy.mean().item()
        v = float(seq_entropy.var().item()) if seq_entropy.numel() > 1 else 0.0
        if self._ema_mean is None:
            self._ema_mean, self._ema_var = m, max(v, 0.1)
        else:
            a = self.ema_alpha
            self._ema_mean = a * self._ema_mean + (1 - a) * m
            self._ema_var  = a * self._ema_var  + (1 - a) * max(v, 0.1)
        self._count += 1

    def thresholds(self):
        """Return (lo, hi) entropy thresholds; (0, inf) during warmup."""
        if self._ema_mean is None or self._count < self.warmup:
            return 0.0, float("inf")
        std = self._ema_var ** 0.5
        return (self._ema_mean - self.lo_sigma * std,
                self._ema_mean + self.hi_sigma * std)


class DistillationController:
    """
    Adapts token-level keep_ratio based on train/val divergence.
    val/train > collapse_threshold → relax (risk of overfitting to hard tokens)
    val/train < recovery_threshold → tighten (model is learning well)
    """
    def __init__(self, initial_keep_ratio=0.70, min_keep_ratio=0.30,
                 max_keep_ratio=1.00, collapse_threshold=1.25,
                 recovery_threshold=1.05, ema_alpha=0.95):
        self.keep_ratio         = initial_keep_ratio
        self.min_keep_ratio     = min_keep_ratio
        self.max_keep_ratio     = max_keep_ratio
        self.collapse_threshold = collapse_threshold
        self.recovery_threshold = recovery_threshold
        self.ema_alpha          = ema_alpha
        self._ema_train = None
        self._ema_val   = None

    def observe_train(self, loss: float):
        if self._ema_train is None:
            self._ema_train = loss
        else:
            a = self.ema_alpha
            self._ema_train = a * self._ema_train + (1 - a) * loss

    def observe_val(self, loss: float):
        self._ema_val = loss

    def step(self) -> float:
        if self._ema_train is None or self._ema_val is None:
            return self.keep_ratio
        ratio = self._ema_val / (self._ema_train + 1e-8)
        if ratio > self.collapse_threshold:
            self.keep_ratio = min(self.max_keep_ratio, self.keep_ratio + 0.05)
            print0(f"  [distill] collapse risk (v/t={ratio:.3f}) → keep={self.keep_ratio:.2f}")
        elif ratio < self.recovery_threshold:
            self.keep_ratio = max(self.min_keep_ratio, self.keep_ratio - 0.01)
        return self.keep_ratio


class DistillationDataLoader:
    """
    Async Smart Filtering pipeline with Early Ejection.

    Background thread scores raw batches via a stale CPU snapshot.
    Batches outside the Goldilocks entropy zone are EJECTED before the
    GPU ever sees them — saving real forward/backward compute.

    Adaptive scoring depth (4 levels, chosen by GPU idle time):
      deep       (idle ≤  3 ms) — all N layers
      medium     (idle ≤  8 ms) — N//2 layers
      light      (idle ≤ 20 ms) — N//4 layers  (≥1)
      ultra_light(idle ≤ 50 ms) — 1 layer
      raw        (idle >  50 ms) — pass-through (GPU starving)
    """

    IDLE_MEDIUM_MS      =  3.0
    IDLE_LIGHT_MS       =  8.0
    IDLE_ULTRALIGHT_MS  = 20.0
    IDLE_RAW_MS         = 50.0

    def __init__(self, base_loader, model_config, controller,
                 buffer_size=8, snapshot_interval=64, train_device="cpu"):
        self.base_loader       = base_loader
        self.controller        = controller
        self.snapshot_interval = snapshot_interval
        self.train_device      = train_device

        n = model_config.n_layer
        self._n_deep        = n
        self._n_medium      = max(1, n // 2)
        self._n_light       = max(1, n // 4)
        self._n_ultralight  = 1

        self._cpu_model = GPT(model_config)
        self._cpu_model.eval()
        self._snap_lock  = threading.Lock()
        self._snap_ready = False

        self._goldilocks = GoldilocksFilter()

        # Stats (main-thread readable, GIL-safe scalars)
        self.idle_ms_ema           = 0.0
        self.idle_ms_total         = 0.0
        self.idle_count            = 0
        self.scoring_mode          = "raw"
        self.avg_keep_ratio        = 1.0
        self.batches_per_output    = 1.0   # raw batches consumed per GPU batch

        self._q    = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="distill-worker")
        self._thread.start()

    # ── main-thread API ───────────────────────────────────────────────────

    def update_snapshot(self, model: nn.Module, step: int):
        if step % self.snapshot_interval != 0:
            return
        raw = model.module if hasattr(model, "module") else model
        cpu_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
        with self._snap_lock:
            self._cpu_model.load_state_dict(cpu_state)
            self._snap_ready = True

    def next_batch(self):
        t0 = time.perf_counter()
        x, y = self._q.get()
        idle_ms = 1000.0 * (time.perf_counter() - t0)

        self.idle_ms_ema   = 0.90 * self.idle_ms_ema + 0.10 * idle_ms
        self.idle_ms_total += idle_ms
        self.idle_count    += 1

        ema = self.idle_ms_ema
        if ema > self.IDLE_RAW_MS:
            self.scoring_mode = "raw"
            self.controller.keep_ratio = min(self.controller.max_keep_ratio,
                                             self.controller.keep_ratio + 0.10)
        elif ema > self.IDLE_ULTRALIGHT_MS:
            self.scoring_mode = "ultra_light"
            self.controller.keep_ratio = min(self.controller.max_keep_ratio,
                                             self.controller.keep_ratio + 0.05)
        elif ema > self.IDLE_LIGHT_MS:
            self.scoring_mode = "light"
            self.controller.keep_ratio = min(self.controller.max_keep_ratio,
                                             self.controller.keep_ratio + 0.02)
        elif ema > self.IDLE_MEDIUM_MS:
            self.scoring_mode = "medium"
        else:
            self.scoring_mode = "deep"

        return x, y

    def reset(self):
        self.base_loader.reset()
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    # ── background worker ─────────────────────────────────────────────────

    def _worker(self):
        consumed = 0
        produced = 0
        while not self._stop.is_set():
            x, y = None, None
            try:
                x, y  = self.base_loader.next_batch()
                mode  = self.scoring_mode if self._snap_ready else "raw"
                result = self._score_and_filter(x, y, mode)
                consumed += 1

                if result is None:
                    # Early Ejection: batch outside Goldilocks zone → skip GPU entirely
                    continue

                produced += 1
                self.batches_per_output = consumed / max(1, produced)
                self._q.put(result)
            except Exception:
                import traceback; traceback.print_exc()
                if x is not None:
                    try: self._q.put((x, y))
                    except Exception: pass

    def _n_layers(self, mode: str) -> int:
        return {
            "deep":        self._n_deep,
            "medium":      self._n_medium,
            "light":       self._n_light,
            "ultra_light": self._n_ultralight,
        }.get(mode, self._n_deep)

    def _score_and_filter(self, x: torch.Tensor, y: torch.Tensor, mode: str):
        """
        Returns (x, y_masked) if batch is in Goldilocks zone, else None.

        Two-level filtering:
          1. Sequence-level: eject sequences outside Goldilocks entropy range.
          2. Token-level: mask easy tokens within kept sequences (one-sided —
             only drop low-entropy tokens; high entropy is always kept since
             that is the definition of the Goldilocks zone).
        """
        if mode == "raw" or not self._snap_ready:
            return x, y

        n = self._n_layers(mode)
        with self._snap_lock:
            with torch.no_grad():
                h = self._cpu_model.transformer.wte(x)
                for blk in list(self._cpu_model.transformer.h)[:n]:
                    h = blk(h)
                h = rmsnorm(h)
                logits = self._cpu_model.lm_head(h)  # (B, T, V)

        probs   = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(-1)  # (B, T)

        # ── 1. Sequence-level Goldilocks ──────────────────────────────────
        seq_ent = entropy.mean(dim=-1)          # (B,)
        self._goldilocks.update(seq_ent)
        lo, hi  = self._goldilocks.thresholds()
        keep_seq = (seq_ent >= lo) & (seq_ent <= hi)

        if not keep_seq.any():
            return None  # Early Ejection — entire batch outside zone

        # ── 2. Token-level masking within kept sequences ──────────────────
        keep_ratio = self.controller.keep_ratio
        y_d = y.clone()

        for i in range(x.size(0)):
            if not keep_seq[i]:
                y_d[i] = -1   # ejected sequence: mask all (won't affect loss)
                continue
            if keep_ratio < 1.0:
                tok_ent = entropy[i]              # (T,)
                n_keep  = max(1, int(keep_ratio * tok_ent.numel()))
                k_rank  = tok_ent.numel() - n_keep
                if k_rank > 0:
                    lo_tok = torch.kthvalue(tok_ent, k_rank).values.item()
                    y_d[i][tok_ent <= lo_tok] = -1  # mask easy tokens

        n_kept = keep_seq.float().mean().item()
        self.avg_keep_ratio = 0.95 * self.avg_keep_ratio + 0.05 * n_kept
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
    parser.add_argument("--batch_size",              type=int,   default=4)
    parser.add_argument("--grad_accumulation_steps", type=int,   default=1)
    parser.add_argument("--sequence_length",         type=int,   default=64)
    # Training
    parser.add_argument("--num_iterations",  type=int,   default=10)
    parser.add_argument("--learning_rate",   type=float, default=1e-4,
                        help="AdamW LR (embeddings + lm_head)")
    parser.add_argument("--muon_lr",         type=float, default=0.02,
                        help="Muon LR (transformer weight matrices)")
    parser.add_argument("--no_muon",         action="store_true",
                        help="Disable Muon, use single AdamW")
    parser.add_argument("--warmup_iters",    type=int,   default=0)
    parser.add_argument("--warmdown_iters",  type=int,   default=0)
    parser.add_argument("--weight_decay",    type=float, default=0.0)
    # Multi-token prediction
    parser.add_argument("--mtp",             type=int,   default=1,
                        help="Multi-token prediction depth (1=off, 2=predict next 2 tokens, …)")
    # Evaluation
    parser.add_argument("--val_loss_every",  type=int,   default=0)
    parser.add_argument("--val_batch_size",  type=int,   default=16)
    parser.add_argument("--save_every",      type=int,   default=256)
    parser.add_argument("--resume",          type=str,   default="",
                        help="Path to checkpoint .pt to resume from")
    # Distillation
    parser.add_argument("--distill",                   action="store_true")
    parser.add_argument("--distill_keep_ratio",        type=float, default=0.70)
    parser.add_argument("--distill_min_keep",          type=float, default=0.30)
    parser.add_argument("--distill_buffer_size",       type=int,   default=8)
    parser.add_argument("--distill_snapshot_interval", type=int,   default=64)
    parser.add_argument("--distill_collapse_threshold",type=float, default=1.25)
    # Misc
    parser.add_argument("--compile",         action="store_true", help="torch.compile (CUDA only)")
    parser.add_argument("--log_wandb",       action="store_true")
    parser.add_argument("--device",          type=str,   default="")
    parser.add_argument("--target_val_loss", type=float, default=0.0,
                        help="Stop early when val_loss ≤ this value (0 = disabled)")
    parser.add_argument("--grad_clip",       type=float, default=1.0,
                        help="Gradient clipping norm (0 = disabled)")
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length

    # ── Device ────────────────────────────────────────────────────────────
    device      = args.device.lower() if args.device else detect_device()
    device_type = device.split(":")[0]
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

    # ── wandb ─────────────────────────────────────────────────────────────
    if args.log_wandb and master_process:
        import wandb, datetime
        wandb.init(
            project="benchmark_gpt2",
            name=f"edsd-muon-{args.model}-{datetime.datetime.now():%m%d-%H%M}",
        )
        wandb.config.update(args)

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # ── Autocast + inductor ───────────────────────────────────────────────
    ctx = make_autocast_ctx(device_type)
    if device_type == "cuda":
        # TF32: free ~2x matmul throughput on Ampere+ with negligible precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
    model_cfgs = {
        "d12": GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768,  mtp_n=args.mtp),
        "d24": GPTConfig(vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, mtp_n=args.mtp),
        "d36": GPTConfig(vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, mtp_n=args.mtp),
        "d48": GPTConfig(vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, mtp_n=args.mtp),
    }
    model_config = model_cfgs[args.model]
    model = GPT(model_config).train().to(device)

    if args.compile and device_type == "cuda":
        print0("compiling the model...")
        model = torch.compile(model, mode="reduce-overhead")

    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if use_ddp else model

    # ── Optimizers ────────────────────────────────────────────────────────
    use_muon = not args.no_muon
    if use_muon:
        muon_opt, adamw_opt = raw_model.make_optimizers(
            muon_lr=args.muon_lr,
            adamw_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        optimizers = [muon_opt, adamw_opt]
        print0(f"optimizer: Muon(lr={args.muon_lr}) + AdamW(lr={args.learning_rate})")
    else:
        adamw_opt = torch.optim.AdamW(
            raw_model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(0.9, 0.95),
        )
        optimizers = [adamw_opt]
        print0(f"optimizer: AdamW(lr={args.learning_rate})")

    if args.mtp > 1:
        print0(f"multi-token prediction: depth={args.mtp}")

    # ── LR schedule ───────────────────────────────────────────────────────
    def get_lr_ratio(it: int) -> float:
        """Multiplier ∈ [0, 1] applied to base LR of each optimizer."""
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        else:
            return (args.num_iterations - it) / args.warmdown_iters

    # Base LRs per optimizer (for schedule scaling)
    opt_base_lrs = []
    for opt in optimizers:
        opt_base_lrs.append([pg["lr"] for pg in opt.param_groups])

    def set_lrs(ratio: float):
        for opt, base_lrs in zip(optimizers, opt_base_lrs):
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = base_lr * ratio

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
        print0(f"[distill] ENABLED  keep_ratio={args.distill_keep_ratio:.2f}  "
               f"snapshot_every={args.distill_snapshot_interval}  "
               f"buffer={args.distill_buffer_size}")
    else:
        print0("[distill] DISABLED — baseline mode")

    # ── Resume ────────────────────────────────────────────────────────────
    start_step       = 0
    training_time_ms = 0.0

    if args.resume:
        print0(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt.get("optimizer_states", [])):
            opt.load_state_dict(state)
        start_step       = ckpt["step"] + 1
        training_time_ms = ckpt.get("training_time_ms", 0.0)
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"])
        print0(f"  resumed at step {start_step}, "
               f"wall-time so far: {training_time_ms/1000:.1f}s")

    # ── Logging ───────────────────────────────────────────────────────────
    run_id  = str(uuid.uuid4())
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, f"{run_id}.log")
        open(logfile, "w").close()

    # ── Training loop ─────────────────────────────────────────────────────
    device_synchronize(device_type)
    t0 = time.perf_counter()

    next_x, next_y = train_loader.next_batch()

    for step in range(start_step, args.num_iterations + 1):
        last_step = step == args.num_iterations

        # ── Validation (always raw FineWeb data) ─────────────────────────
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

            print0(f"step:{step}/{args.num_iterations} | val_loss {val_loss_scalar:.6f} | "
                   f"wall:{training_time_ms/1000:.1f}s")

            if controller is not None:
                controller.observe_val(val_loss_scalar)
                new_ratio = controller.step()
                if isinstance(train_loader, DistillationDataLoader):
                    print0(f"  [distill] keep={new_ratio:.3f}  "
                           f"avg_kept={train_loader.avg_keep_ratio:.3f}  "
                           f"idle_ema={train_loader.idle_ms_ema:.1f}ms[{train_loader.scoring_mode}]")

            if master_process:
                if args.log_wandb:
                    ld = {"val_loss": val_loss_scalar, "wall_s": training_time_ms / 1000}
                    if controller:
                        ld["keep_ratio"] = controller.keep_ratio
                    wandb.log(ld, step=step * tokens_per_iter)
                if logfile:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} val:{val_loss_scalar:.6f}\n")

            # Early stopping: target val_loss reached
            if args.target_val_loss > 0 and val_loss_scalar <= args.target_val_loss:
                print0(f"Target val_loss {args.target_val_loss} reached at step {step}. Stopping.")
                break

            device_synchronize(device_type)
            t0 = time.perf_counter()

        if last_step:
            break

        # ── Snapshot for distillation worker ─────────────────────────────
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
            next_x, next_y = train_loader.next_batch()
            loss.backward()

        ratio = get_lr_ratio(step)
        set_lrs(ratio)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        # ── Diagnostics ───────────────────────────────────────────────────
        device_synchronize(device_type)
        approx_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        if use_ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()

        if controller is not None:
            controller.observe_train(lossf)

        idle_tag = ""
        if isinstance(train_loader, DistillationDataLoader):
            tl = train_loader
            idle_tag = (f" | kept:{tl.avg_keep_ratio:.2f}"
                        f" | consumed:{tl.batches_per_output:.1f}x"
                        f" | idle:{tl.idle_ms_ema:.1f}ms[{tl.scoring_mode}]")

        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | "
            f"lr_r:{ratio:.3f} | "
            f"train_time:{approx_ms/1000:.1f}s | step_avg:{approx_ms/(step - start_step + 1):.0f}ms"
            + idle_tag
        )

        if master_process and logfile:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trn:{lossf:.6f}\n")

        if master_process and (step + 1) % args.save_every == 0:
            elapsed_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            ckpt = dict(
                model            = raw_model.state_dict(),
                optimizer_states = [opt.state_dict() for opt in optimizers],
                step             = step,
                training_time_ms = elapsed_ms,
                rng              = torch.get_rng_state(),
                args             = args.__dict__,
                code             = code,
            )
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            path = f"logs/{run_id}/ckpt_{step:06d}.pt"
            torch.save(ckpt, path)
            torch.save(ckpt, f"logs/{run_id}/latest.pt")
            print0(f"  [ckpt] → {path}")

    # ── Idle-time summary ────────────────────────────────────────────────
    if isinstance(train_loader, DistillationDataLoader) and train_loader.idle_count > 0:
        tl = train_loader
        avg_idle = tl.idle_ms_total / tl.idle_count
        pct = 100.0 * tl.idle_ms_total / max(1.0, training_time_ms + tl.idle_ms_total)
        print0(f"\n[idle-report] total={tl.idle_ms_total/1000:.2f}s  "
               f"avg/call={avg_idle:.2f}ms  pct={pct:.1f}%  "
               f"final_mode={tl.scoring_mode}  final_keep={tl.controller.keep_ratio:.2f}")
        if avg_idle >= DistillationDataLoader.IDLE_RAW_MS:
            print0("  → CPU scorer too slow. Try --distill_snapshot_interval 128 or reduce batch.")
        elif avg_idle >= DistillationDataLoader.IDLE_LIGHT_MS:
            print0("  → Light mode active. Worker marginal — snapshot_interval could increase.")
        else:
            print0("  → Worker kept up. Full scoring throughout — distillation maximally effective.")

    # ── Peak memory ───────────────────────────────────────────────────────
    if device_type == "cuda":
        print0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    elif device_type == "mps":
        try:
            print0(f"peak memory: {torch.mps.driver_allocated_memory() // 1024 // 1024} MiB")
        except Exception:
            pass

    # ── Final checkpoint ─────────────────────────────────────────────────
    if master_process:
        ckpt = dict(
            model            = raw_model.state_dict(),
            optimizer_states = [opt.state_dict() for opt in optimizers],
            step             = args.num_iterations,
            training_time_ms = training_time_ms,
            rng              = torch.get_rng_state(),
            args             = args.__dict__,
            code             = code,
        )
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch.save(ckpt, f"logs/{run_id}/final.pt")
        torch.save(ckpt, f"logs/{run_id}/latest.pt")
        print0(f"done. Final checkpoint: logs/{run_id}/final.pt")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if isinstance(train_loader, DistillationDataLoader):
        train_loader.stop()
    if use_ddp:
        destroy_process_group()
