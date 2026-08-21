import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2}, num_stages=2),
    ],
    key=["N_CTX"],
)
@triton.jit
def _flash_attention_fwd(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptrs + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    kT_ptrs = k_ptrs + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    vT_ptrs = v_ptrs + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for j in range(N_CTX // BLOCK_N):
        kT = tl.load(kT_ptrs)
        qk = tl.dot(q, kT) * sm_scale
        if CAUSAL:
            causal_mask = offs_m[:, None] >= (j * BLOCK_N + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp(qk - m_ij[:, None]).to(tl.float16)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vT = tl.load(vT_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p, vT)
        m_i = m_ij
        kT_ptrs += BLOCK_N * stride_kn
        vT_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh
    tl.store(out_ptrs + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc.to(tl.float16))


def attention(q, k, v, causal=False, sm_scale=None):
    assert q.ndim == 4
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
    _flash_attention_fwd[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    return o
