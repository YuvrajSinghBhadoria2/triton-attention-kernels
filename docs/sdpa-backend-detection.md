# Why SDPA is slow at decode: silent MATH fallback

This note explains the mechanism behind the long-context decode results in
[`long-context-decode.md`](long-context-decode.md). It is not a new benchmark — it is a
controlled backend-isolation run on the same Kaggle T4 that answers two questions:

1. Why does PyTorch SDPA get progressively slower as decode context grows?
2. How can a caller detect that SDPA silently fell back to the MATH reference path?

## Environment

- Kaggle Tesla T4 (sm75)
- torch 2.11.0+cu130, triton 3.6.0
- Shape taken from [vLLM #66515](https://github.com/vllm-project/vllm/issues/66515):
  `B=32, H_Q=32, H_KV=8, D=64, ctx=2048, Q len=1, fp16`

## Method

PyTorch's `torch.nn.attention.sdpa_kernel(SDPBackend.X)` context manager restricts SDPA to a
single backend. If the shape is unsupported, it raises
`RuntimeError: No available kernel. Aborting execution.` Running the same decode call under each
backend therefore tells us exactly which kernels accept the shape, without guessing from
timings.

## Result

| path | FLASH | MEM_EFFICIENT | MATH | default |
|---|---|---|---|---|
| `enable_gqa=True` (grouped-query, no materialization) | no kernel | no kernel | 17.82 ms | **16.81 ms** |
| materialized KV (`repeat_interleave`, H_Q/H_KV=4) | no kernel | 2.37 ms | 2.79 ms | **2.53 ms** |

At `Q len = 1`, the fused FLASH and MEM_EFFICIENT kernels do not accept the grouped-query
shape, so `enable_gqa=True` has only MATH available. The materialized path qualifies for
mem-efficient and is ~6.6x faster in the default configuration. This is the same mechanism
reported independently in vLLM #66515 (their ratio was 2.59x; ours is larger, likely due to
shape and timing-loop differences).

This is also the reason the split-KV kernel in this repo wins: it bypasses SDPA's backend
selection entirely and uses a Triton kernel designed for `Q len = 1`, where the work is
memory-bound over the KV cache rather than a matrix-multiply-shaped prefill problem.

## Detecting the fallback programmatically

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

for backend in (SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH):
    try:
        with sdpa_kernel(backend):
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        print(backend, "available")
    except RuntimeError as e:
        print(backend, "rejected shape:", e)
```

If MATH is the only backend that survives, the call is running the reference path. Checking
the kernel name under `torch.profiler` is an equivalent alternative.

## Implication for inference engines

An engine that relies on SDPA for decode with GQA should either materialize KV to reach
mem-efficient, or use a dedicated decode kernel (split-KV / flash-decoding). Silently landing
on MATH at decode is a large, shape-dependent regression that will not show up in prefill
benchmarks.

Reproduction data and the backend-forcing script are in the Kaggle notebook linked from the
main [README](../README.md).
