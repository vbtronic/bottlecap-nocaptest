# Smart Filtering — Goldilocks Zone + Early Ejection

*Thanks to Ondřej Plátek for the key observation: masking tokens in the loss still pays for the full
forward/backward pass, so real savings only come from data that never reaches the GPU.*

## Motivation

Not every sequence in FineWeb is equally useful for a 124M model:

- **Too easy** — boilerplate, repeated navigation text, things the model already predicts well. Small loss, small gradient.
- **Too hard** — tables, minified code, other languages, spam. High loss that is mostly irreducible noise.

The budget allows 5B training tokens while the baseline trains on 2.5B. Smart Filtering spends part of that
spare budget on *reading* more data and only *trains* on the sequences in the middle — the Goldilocks zone.

## Method

For every micro-batch of `B` sequences:

1. Read a candidate pool of `ceil(B * filter_pool)` sequences (default 1.25 → 20 candidates for B = 16).
2. Score each candidate with the **live** training model under `no_grad`: mean cross-entropy over the first
   `filter_score_len` tokens (default 256 of 1024).
3. Sort by score and eject the easiest `(1 - hard_frac)` and hardest `hard_frac` share of the surplus
   (default 2 + 2 sequences).
4. Train on the remaining `B` sequences exactly as the baseline does.

Scoring cost: a forward pass is ~1/3 of forward+backward, and scoring touches 1.25 × 1/4 of the tokens,
so the overhead is roughly 1.25 × 1/4 × 1/3 ≈ **10% per step**. Scoring time is counted as training time.
The filter is off for the first `filter_warmup` steps (default 128), while the model's scores are still noise.
Validation always uses the unfiltered validation set.

Data use with the defaults: 128 × 0.52M + 4640 × 0.66M ≈ 3.1B tokens read, within the 5B budget.

For the method to win, the step-for-step improvement from better data has to beat the ~10% slowdown per step.

## History — what the first version did wrong

The first version (April 2026, commits `0882d9a`..`8870eb5`) had the same goal but did not work:

| Problem | Effect |
|---|---|
| Scored with a full 124M copy of the model **on the CPU**, building `(B, T, 50257)` fp32 logits (~3 GB per batch) | The CPU is orders of magnitude slower than the GPU it was supposed to feed. The scorer fell back to "raw" pass-through, so in practice nothing was filtered. |
| "Early Ejection" dropped a batch only when *all* sequences were out of the zone; otherwise out-of-zone sequences were masked with `-1` | The GPU still ran the forward/backward on them. No compute was saved. |
| Token-level masking of the lowest-entropy 30% of tokens | Again no compute saved; it also shrinks the effective batch. Low entropy ≠ low loss: confidently wrong tokens, the most informative ones, were thrown away. |
| Collapse controller compared val loss with train loss computed on *hard* tokens only | Train loss is always higher than val loss there, so the ratio never triggered and `keep_ratio` just decayed to the minimum. |
| Snapshot lock held for the whole CPU forward | The training loop blocked on `update_snapshot`. |
| Every raw-mode batch increased `keep_ratio` by 0.10 | The filter switched itself off. |
| Muon optimizer and multi-token prediction heads on by default in `run_mps.sh` | Muon is a modded-nanogpt speedup (not allowed); MTP heads added 2 × 38M parameters and caused the MPS OOM. |
| No `torch.compile` in our run while the baseline compiles | Unfair comparison against ourselves. |
| A ~1.5 GB checkpoint every 256 steps; resuming did not restore the data loader position | Fills the disk; a resumed run retrained on data it had already seen. |

## Results

Pending — run `colab.ipynb` (baseline and Smart Filtering on the same GPU).

| Run | GPU | Steps | Time to val ≤ 3.3821 | Tokens read |
|---|---|---|---|---|
| baseline | | | | |
| Smart Filtering | | | | |
