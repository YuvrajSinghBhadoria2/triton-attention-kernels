# Why Split-KV Decode Beats PyTorch SDPA by 4× at 64K Context on a T4

**TL;DR:** Q=1 decode attention is memory-bound: one query token scans the entire KV cache. PyTorch SDPA assigns one program per (batch, head), so at long context a single program serially streams all 64K keys — the GPU is mostly idle. Splitting the KV dimension across 64 Triton programs and reducing partial softmax results in a second pass makes the same math **4.03× faster** than SDPA at 64K on an NVIDIA T4, with `allclose` correctness.

All numbers are reproducible: [Kaggle T4 notebook](https://www.kaggle.com/code/yuvrajsinghbhadoria/long-context-decode-benchmark) · [code](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels).

## The problem

Decode attention has query length 1. For each (batch, head), you compute:

```
softmax(q · Kᵀ / √d) · V
```

where K, V are `[n_ctx, d]`. At n_ctx = 65536 this reads ~16 MB of KV per head. A standard attention kernel launches one CTA per (batch, head). On a T4 with 20 SMs and B=1, H=8, that is 8 CTAs — **less than half the SMs are used**, and each CTA does 64K iterations of serial KV streaming.

This is the classic decode bottleneck: low occupancy, memory-bound, and no parallelism left in the query dimension to exploit.

## The fix: split-KV

Split the KV cache into `S` chunks. Each program processes one chunk and writes a **partial output** plus a **partial log-sum-exp (LSE)**:

```
partial_out[s] = softmax_local(q · K[s]ᵀ / √d) · V[s]
partial_lse[s] = log-sum-exp of q · K[s]ᵀ / √d
```

Then combine with the standard online-softmax merge:

```
w[s]      = exp(lse[s] - max(lse))
out       = Σ w[s] · partial_out[s] / Σ w[s]
```

Each split is an independent CTA, so 64 splits means 64 × 8 = 512 CTAs on an 8-head workload — full occupancy, KV streamed in parallel.

**Correctness detail that matters:** to avoid `exp(-inf - -inf) → NaN` in the online-softmax update, the kernel skips fully-empty tiles and uses `l_i > 0` guards. It also aligns each split to a whole number of `BLOCK_N` tiles so the masked `tl.dot` path (which diverges numerically on this target) is never taken. Every configuration below is `allclose` to SDPA at atol/rtol 0.02.

## Measured results — NVIDIA T4, fp16, B=1, H=8, D=64

10 warmup + 50 timed CUDA-event iterations, best split count per context:

| Context | Best splits | Triton (ms) | SDPA (ms) | Speedup | Correct |
|---:|---:|---:|---:|---:|:---:|
| 1K  | 1  | 0.232 | 0.166 | 0.72× (slower) | ✅ |
| 2K  | 4  | 0.224 | 0.324 | 1.45× | ✅ |
| 4K  | 16 | 0.245 | 0.643 | 2.62× | ✅ |
| 8K  | 64 | 0.229 | 1.019 | 4.45× | ✅ |
| 16K | 64 | 0.294 | 1.086 | 3.69× | ✅ |
| 32K | 64 | 0.547 | 2.162 | 3.95× | ✅ |
| 64K | 64 | 1.076 | 4.340 | **4.03×** | ✅ |

## What the numbers actually say

**1. Short context is launch-bound.** At 1K, split-KV (0.232 ms) loses to SDPA (0.166 ms). There is not enough work per split to amortize the extra kernel launch and the partial-output reduction. Splitting is not free.

**2. The crossover is ~2K.** From 2K onward, the parallelism win dominates. At 4K the split kernel is already 2.6× faster.

**3. Long context is where it compounds.** At 64K, SDPA's single program per head streams the whole cache serially (4.34 ms). The 64-split kernel finishes in 1.08 ms. The speedup grows with context length — exactly the regime that matters for long-context LLM serving.

**4. More splits help at long context (contrary to the short-context result).** In the earlier 1K–16K study, "128 splits was worst everywhere." Here, 64 splits is best at every context ≥8K. The right split count is a function of context length, not a fixed constant — a useful production tuning insight.

## Why this matters for LLM inference

Decode phase dominates inter-token latency (TPOT) at long context. A 4× reduction in attention kernel time directly reduces TPOT and raises SLO-goodput for long-context workloads — the exact bottleneck in document analysis, agentic tool use with long histories, and RAG with large retrieved contexts.

The kernel is pure Triton (no CUDA C++), so it is portable across NVIDIA and AMD, and it composes with paged KV-cache layouts used by vLLM/SGLang.

## Reproduce

```bash
git clone https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels.git
# Open long_context_decode_benchmark.ipynb on Kaggle with GPU T4 → Run All
```

Environment used: Kaggle T4, torch 2.11.0+cu130, triton 3.6.0.

## Honest caveats

- B=1, H=8, D=64. Larger batches change the occupancy math and the crossover point.
- Compared against PyTorch SDPA on T4 (sm75), which is vendor-tuned but not the fastest possible attention implementation on that GPU.
- The 0.02 atol/rtol correctness check is appropriate for fp16 attention but is not a bit-exact guarantee.
- Speedup is context-dependent: do not quote "4×" for 1K context.
