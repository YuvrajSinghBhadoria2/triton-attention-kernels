import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, KV_Cache, BlockTable, Out, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kn, stride_kps, stride_kkv, stride_kh, stride_kd,
    stride_bb, stride_bm,
    stride_ob, stride_oh, stride_od,
    B, H_Q, H_KV, D, PAGE_SIZE, MAX_BLOCKS, N_CTX,
    BLOCK_DMODEL: tl.constexpr,
):
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // (H_Q // H_KV)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q = tl.load(Q + b * stride_qb + h_q * stride_qh + offs_d * stride_qd).to(tl.float32) * sm_scale

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for p in range(0, N_CTX):
        page = tl.load(BlockTable + b * stride_bb + (p // PAGE_SIZE) * stride_bm)
        off_in_page = p % PAGE_SIZE
        k = tl.load(KV_Cache + page * stride_kn + off_in_page * stride_kps
                    + 0 * stride_kkv + h_kv * stride_kh + offs_d * stride_kd).to(tl.float32)
        v = tl.load(KV_Cache + page * stride_kn + off_in_page * stride_kps
                    + 1 * stride_kkv + h_kv * stride_kh + offs_d * stride_kd).to(tl.float32)
        qk = tl.sum(q * k)
        m_ij = tl.maximum(m_i, qk)
        p_ = tl.math.exp(qk - m_ij)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + p_
        acc = acc * alpha + p_ * v
        m_i = m_ij
    acc = acc / l_i
    tl.store(Out + b * stride_ob + h_q * stride_oh + offs_d * stride_od, acc.to(tl.float16))


def paged_attention(q, kv_cache, block_table, n_ctx, sm_scale=None):
    # kv_cache: (NUM_PAGES, PAGE_SIZE, 2, H_KV, D)  -> idx 0 = K, idx 1 = V
    B, H_Q, _, D = q.shape
    NUM_PAGES, PAGE_SIZE, _, H_KV, D = kv_cache.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    assert q.shape[2] == 1
    assert H_Q % H_KV == 0
    o = torch.empty_like(q).squeeze(2)
    grid = (B, H_Q)
    _paged_decode_kernel[grid](
        q.squeeze(2), kv_cache, block_table, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        kv_cache.stride(3), kv_cache.stride(4),
        block_table.stride(0), block_table.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        B, H_Q, H_KV, D, PAGE_SIZE, block_table.shape[1], n_ctx,
        BLOCK_DMODEL=D, num_warps=4,
    )
    return o.unsqueeze(2)
