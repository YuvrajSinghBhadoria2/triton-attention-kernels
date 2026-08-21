# Portable Triton Attention Kernels — Prefill, Split-KV Decode, Paged (T4 / Colab)

A from-scratch, portable Triton implementation of the three attention kernels that
power LLM inference — fused prefill, split-KV decode, and paged KV-cache attention —
built to demonstrate production-style inference attention on constrained hardware.

## What's inside

| File | Kernel | Why it matters |
|------|--------|----------------|
| `kernels/flash_attention.py` | Fused prefill attention (FA2-style, online softmax) | Baseline fused kernel; proves correctness methodology |
| `kernels/flash_decoding.py` | **Split-KV flash-decoding** (decode, `Q=1`) | Headline: parallelizes the KV dimension across SMs to fix the memory-bound decode path |
| `kernels/paged_attention.py` | Paged decode over block tables (GQA-aware) | The inference-systems piece: portable paged-KV attention like vLLM's Triton backend |

All kernels are **portable Triton implementations** (no CUDA C++), fp16, and run on
NVIDIA T4 (Colab free tier, sm75). No TMA/FP8/warp-spec — those need Hopper+ and
are intentionally out of scope for this target.

## Run in Google Colab (free T4) — recommended: the notebook

1. Open `triton_attention_colab.ipynb` in Colab (Upload, or `Open from GitHub` after
   you fork).
2. Runtime → Change runtime type → **GPU** (T4).
3. Runtime → Run all. It runs correctness checks, then the prefill-memory and decode-latency benchmarks.

### Alternative: use the kernel modules directly

`kernels/` is a standalone, importable library. After
`pip install -r requirements.txt`:

```python
import torch
from kernels.flash_attention import attention
from kernels.flash_decoding import flash_decoding
from kernels.paged_attention import paged_attention

out = attention(q, k, v, causal=True)          # prefill (fused, FA2-style)
out = flash_decoding(q, k, v, num_splits=32)   # decode (split-KV over the KV cache)
out = paged_attention(q, kv_cache, block_table, n_ctx)  # paged decode (block-table)
```

The full correctness checks and the prefill-memory / decode-latency benchmarks
live in `triton_attention_colab.ipynb`.

## Correctness rule

Every kernel is checked against
`torch.nn.functional.scaled_dot_product_attention` with
`torch.allclose(atol=1e-2, rtol=1e-2)` for fp16, across prefill/decode/paged
shapes (including non-power-of-two and GQA).

## Expected result on T4

- All correctness checks print `True`.
- **Prefill (cell 7):** the honest, demonstrable win is **memory**. Fused Triton
  attention is `O(N·D)`; the naive materialized attention builds the full
  `[B,H,N,N]` score matrix and **OOMs on a 16GB T4 at large N while Triton keeps
   running**. (Peak-memory is measured exactly and is unaffected by Colab's compute
   throttling, so these numbers are trustworthy. On a free T4, PyTorch's SDPA
   dispatches to expert-tuned FlashAttention-2 / cuDNN, so don't expect a raw-speed
   win — that is expected, not a failure.)
- **Decode (cell 8):** performance characterization of split-KV `flash_decoding`
  across `num_splits` ∈ {1,8,16,32,64,128} and N_CTX 1024→16384, reporting kernel
  latency vs an SDPA baseline. Correctness is covered by cell 5's `allclose=True`
  checks. The split-KV value is the *architecture* (parallelizing the KV dimension
  across SMs, the technique vLLM uses for long-context decode), not a raw-speed win
  over SDPA — see the measured table under "Decode latency characterization".


> Numbers vary with T4 contention in free Colab. Peak-memory (cell 7) is exact;
> latency (cell 8) is only comparable across rows run in the same session. Run 2–3
> times; report the median. Shapes are kept multiples of 64 (the kernel's block
> size) — add edge masking for arbitrary shapes as a follow-up.

## Decode latency characterization (measured, NVIDIA T4)

Benchmarked `flash_decoding` across `N_CTX` ∈ {1024, 2048, 4096, 8192, 16384} and
`num_splits` ∈ {1, 8, 16, 32, 64, 128}, with 20 warmup + 100 measured CUDA-event
iterations (B=4, H=16, D=64, Q=1, fp16). SDPA is a reference baseline only.

**Best `num_splits` per context (lowest kernel latency):**

| N_CTX | best num_splits | Triton (ms) | SDPA (ms) | Triton / SDPA |
|------:|----------------:|------------:|----------:|--------------:|
|  1024 |              32 |       0.405 |     0.249 |         1.62× |
|  2048 |              32 |       0.297 |     0.379 |         0.78× |
|  4096 |              16 |       0.528 |     0.393 |         1.34× |
|  8192 |              16 |       1.018 |     0.845 |         1.21× |
| 16384 |              16 |       2.012 |     1.737 |         1.16× |

**Observations.**
- Splitting the KV dimension sharply cuts decode latency versus a single split
  (e.g. N=1024: 1.14 ms at `num_splits=1` → 0.40 ms at `num_splits=32`). With
  `Q_LEN=1` there is no query-dimension parallelism, so split-KV is the only source
  of sequence parallelism for decode.
- More splits is *not* monotonic: `num_splits=128` is the worst or near-worst
  configuration at every context length (launch + reduction overhead dominates),
  and the sweet spot (16–32) does not increase steadily with context.
- Versus SDPA the best Triton split is within ~1.2–1.6× at 4K–16K and marginally
  faster at 2K (0.78×) in this run; SDPA remains the vendor-tuned reference. The
  benchmark's purpose is characterizing split-KV scaling, not a speed claim.

## Conclusion

**What this project delivers: three Triton attention kernels covering the full LLM serving path, each a standalone file and each numerically correct against PyTorch SDPA (fp16, `allclose=True`) from 1K to 16K tokens:**

- **Fused prefill attention** (`flash_attention.py`) — processes the whole input prompt in one pass (FlashAttention-2-style online softmax, autotuned).
- **Split-KV flash-decoding** (`flash_decoding.py`) — generates one token at a time; split-KV parallelizes the work across the KV cache so the tiny Q=1 matmul stays fast.
- **Paged KV-cache attention** (`paged_attention.py`) — reads the KV cache through a block table, the memory layout real serving engines use; a capability SDPA *cannot* express.

**What the measurements prove**

| Finding | Measured result |
|---------|-----------------|
| Prefill memory vs naive | **10.2× less** @4K (0.44 vs 4.46 GB); naive **OOMs @8K**, Triton fits @0.57 GB |
| Decode gain from split-KV | split 1→32 @1K: **1.14 → 0.40 ms** |
| Best decode split | **16–32** (does *not* grow with N); `num_splits=128` worst at every context |
| Decode vs SDPA | 0.78× @2K, 1.2–1.6× @4K–16K (SDPA is vendor-tuned FA2/cuDNN) |

**Bottom line.** A verified, portable Triton attention library — correct against a
vendor reference, memory-efficient on a constrained GPU, and extending to paged KV
that PyTorch's built-in attention leaves to the inference engine.

## Status

Validated and benchmarked on an NVIDIA T4 (Colab free tier, sm75), fp16. All kernels
pass `allclose` against PyTorch SDPA; prefill peak-memory and decode split-KV latency
are characterized in the sections above.
