# LinkedIn Post — Triton Attention Kernels

> Format: text post + 2 attached screenshots. Link goes in the first comment (LinkedIn
> penalizes links in the body). Post the screenshots of the actual benchmark runs.

---

I built three portable Triton attention kernels for LLM inference and benchmarked them on an NVIDIA T4 (Colab free tier).

Screenshots of the actual benchmark runs are attached.

What's in the repo:

→ Fused prefill attention (FlashAttention-2 style, online softmax)
→ Split-KV flash-decoding for single-token decode
→ Paged KV-cache attention with block-table lookup (like vLLM)

All three are verified `allclose` against PyTorch's fused SDPA across 1K–16K contexts.

What the benchmarks show:

→ Prefill memory: 10.2× less peak memory than naive at 4K (0.44 vs 4.46 GB); naive OOMs at 8K while Triton runs at 0.57 GB
→ Decode split-KV: sweeping num_splits 1→128, the best split was 16–32 and num_splits=128 was the worst at every context length
→ Portable Triton throughout (no CUDA C++), runs on sm75

The paged kernel does what PyTorch's SDPA cannot: read KV through block tables — the memory layout real serving engines use.

Full repo + Colab notebook in the comments 👇

What would you optimize first — memory, decode latency, or coverage?

#Triton #LLMInference #GPU #PyTorch #MachineLearning

---

## Attach these screenshots (from the notebook runs)

1. **Prefill memory table** — naive vs Triton GB across 512→8192, OOM at 8K.
2. **Decode latency characterization** — N_CTX × num_splits dependence table + best-split analysis.

## First comment

`https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels`
