# Triton Attention Kernels — Prefill, Split-KV Decode, Paged KV-Cache

<p>
  <img src="https://img.shields.io/badge/Triton-portable%20kernels-76CC4D" alt="Triton">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20T4%20%7C%20Colab-76B900" alt="T4">
  <img src="https://img.shields.io/badge/Precision-FP16-orange" alt="FP16">
  <img src="https://img.shields.io/badge/Correctness-SDPA%20allclose%3DTrue-blue" alt="SDPA">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

**One-line:** Three portable Triton kernels covering the full LLM serving path — fused prefill, split-KV flash-decoding (Q=1), and paged block-table attention — verified `allclose` vs PyTorch SDPA 1K→16K. On T4: **10.2× less memory @4K (naive OOMs @8K)**, split-KV **1.14→0.40 ms** with optimal split 16–32.

> Built for LLM inference on constrained hardware. No CUDA C++ — pure Triton, fp16, T4 (sm75). Paged KV is the systems piece PyTorch SDPA cannot express.

### What's Inside

| File | Kernel | Why it matters |
|------|--------|----------------|
| `kernels/flash_attention.py` | Fused prefill (FA2-style, online softmax) | Proves correctness methodology |
| `kernels/flash_decoding.py` | **Split-KV flash-decoding** (Q=1) | Parallelizes KV across SMs — fixes memory-bound decode |
| `kernels/paged_attention.py` | Paged decode over block tables (GQA-aware) | Portable paged KV like vLLM's Triton backend |

### Key Results — NVIDIA T4, fp16, B=4, H=16, D=64

| Finding | Measured |
|---------|----------|
| **Prefill memory @4K** | **0.44 GB vs 4.46 GB naive (10.2× less)**; naive **OOMs @8K**, Triton **0.57 GB** |
| **Decode split-KV gain @1K** | **1.14 ms → 0.40 ms** (split 1 → 32) |
| **Best split** | **16–32** (not monotonic; 128 worst everywhere) |
| **Decode vs SDPA** | 0.78× @2K, 1.2–1.6× @4K–16K (SDPA = vendor-tuned FA2/cuDNN) |

<details>
<summary>Decode — Best split per context (lowest latency, 20 warmup + 100 CUDA-event iters)</summary>

| N_CTX | best splits | Triton (ms) | SDPA (ms) | Triton/SDPA |
|------:|------------:|------------:|----------:|------------:|
| 1024 | 32 | 0.405 | 0.249 | 1.62× |
| 2048 | 32 | 0.297 | 0.379 | **0.78×** |
| 4096 | 16 | 0.528 | 0.393 | 1.34× |
| 8192 | 16 | 1.018 | 0.845 | 1.21× |
| 16384 | 16 | 2.012 | 1.737 | 1.16× |

Splitting KV is the only sequence parallelism for Q=1; 128 splits always hurts (launch + reduction overhead).

</details>

### Run in Colab (Free T4) — 1 click

1. Open `triton_attention_colab.ipynb` → Upload to Colab
2. Runtime → GPU (T4) → Run all — correctness, prefill-memory (cell 7), decode-latency (cell 8)

### Use as a Library

```python
import torch
from kernels.flash_attention import attention
from kernels.flash_decoding import flash_decoding
from kernels.paged_attention import paged_attention

out = attention(q, k, v, causal=True)                      # prefill
out = flash_decoding(q, k, v, num_splits=32)               # decode split-KV
out = paged_attention(q, kv_cache, block_table, n_ctx)     # paged block-table
```

`pip install -r requirements.txt` first. Full checks + benchmarks are in the notebook.

### Correctness

Every kernel vs `torch.nn.functional.scaled_dot_product_attention`, `allclose(atol=1e-2, rtol=1e-2)` fp16, prefill/decode/paged, including non-power-of-two and GQA.

> Peak-memory (cell 7) is exact; latency varies with T4 contention — run 2–3× and report median. Shapes are multiples of 64 (kernel block size).

### Status

Validated on T4 sm75, fp16. All kernels pass SDPA allclose 1K→16K; memory and split-KV scaling characterized above.
