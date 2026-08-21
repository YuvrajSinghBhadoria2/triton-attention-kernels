import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    Q, K, V, sm_scale,
    PartialOut, PartialLSE,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pb, stride_ph, stride_ps, stride_pd,
    stride_lb, stride_lh, stride_ls,
    B, H, N_CTX,
    NUM_SPLITS: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    qk_base = b * H + h
    Q_pt = Q + qk_base * stride_qh
    K_base = K + qk_base * stride_kh
    V_base = V + qk_base * stride_vh
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q_pt + offs_d * stride_qd).to(tl.float32) * sm_scale  # [D]

    # Align each split to a whole number of BLOCK_N tiles. This avoids the
    # partially-valid (masked) tl.dot path, which is the only case that diverges
    # from the exact math on this target. Empty trailing splits contribute 0.
    kv_per_split = tl.cdiv(tl.cdiv(N_CTX, BLOCK_N), NUM_SPLITS) * BLOCK_N
    start_n = split * kv_per_split

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for tile in range(0, kv_per_split, BLOCK_N):
        p_start = start_n + tile
        valid = (p_start + offs_n < N_CTX) & (p_start + offs_n < start_n + kv_per_split)  # [BLOCK_N]
        # Skip fully-empty tiles (e.g. trailing splits past N_CTX) to avoid
        # exp(-inf - -inf) -> nan in the online-softmax update.
        if tl.max(valid) != 0:
            k_ptrs = K_base + offs_d[:, None] * stride_kd + (p_start + offs_n)[None, :] * stride_kn
            k = tl.load(k_ptrs, mask=valid[None, :], other=0.0).to(tl.float32)  # [D, BLOCK_N]
            qk = tl.dot(q[None, :], k)  # [1, BLOCK_N]
            qk = tl.where(valid[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk))  # scalar
            p = tl.math.exp(qk - m_ij)  # [1, BLOCK_N]
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p)
            v_ptrs = V_base + (p_start + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=valid[None, :], other=0.0).to(tl.float32)  # [BLOCK_N, D]
            acc = acc * alpha + tl.reshape(tl.dot(p, v), [BLOCK_DMODEL])  # [D]
            m_i = m_ij

    lse_i = tl.where(l_i > 0, m_i + tl.math.log(l_i), -float("inf"))
    acc_out = acc / tl.where(l_i > 0, l_i, 1.0)
    tl.store(PartialOut + qk_base * stride_ph + split * stride_ps + offs_d * stride_pd, acc_out)
    tl.store(PartialLSE + qk_base * stride_lh + split * stride_ls, lse_i)


def flash_decoding(q, k, v, num_splits=64, sm_scale=None):
    B, H, _, D = q.shape
    N_CTX = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    assert q.shape[2] == 1, "flash_decoding is for decode (query seqlen == 1)"
    q = q.squeeze(2)
    partial_out = torch.zeros((B, H, num_splits, D), device=q.device, dtype=torch.float32)
    partial_lse = torch.zeros((B, H, num_splits), device=q.device, dtype=torch.float32)
    grid = (B, H, num_splits)
    _decode_kernel[grid](
        q, k, v, sm_scale, partial_out, partial_lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        B, H, N_CTX, NUM_SPLITS=num_splits, BLOCK_DMODEL=D, BLOCK_N=min(64, D),
        num_warps=4,
    )
    lse_max = partial_lse.max(-1, keepdim=True).values
    w = torch.exp(partial_lse - lse_max)
    out = (partial_out * w.unsqueeze(-1)).sum(2) / w.sum(-1, keepdim=True)
    return out.to(torch.float16).unsqueeze(2)
