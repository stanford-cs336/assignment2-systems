import torch
import math
import triton
import triton.language as tl

@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # l_block_ptr = tl.make_block_ptr(
    #     L_ptr + batch_index * stride_lb,
    #     shape=(N_QUERIES, ),
    #     strides=(stride_lq,),
    #     offsets=(query_tile_index * Q_TILE_SIZE,),
    #     block_shape=(Q_TILE_SIZE,),
    #     order=(0,),
    # )
    l_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    l_ptrs = L_ptr + batch_index * stride_lb + l_offsets * stride_lq
    l_mask = l_offsets < N_QUERIES

    q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero") 
    m = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    l = tl.full([Q_TILE_SIZE], 0, dtype=tl.float32)
    O = tl.full([Q_TILE_SIZE, D], 0, dtype=tl. float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(i * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        v_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(i * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        m_prev = m
        l_prev = l
        O_prev = O
        k_tile = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero") 
        v_tile = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")

        S = tl.dot(q_tile, tl.trans(k_tile))*scale
        if IS_CAUSAL:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S = tl.where(causal_mask, S, float("-inf"))
        m_new = tl.maximum(m_prev, triton.language.max(S, axis=1))
        P = tl.exp(S - m_new[:, None])                                           
        l_new = tl.exp(m_prev-m_new) * l_prev + tl.sum(P, axis=1)
        scale_vector = tl.exp(m_prev - m_new)
        scaled_O_prev = scale_vector[:, None] * O_prev
        O_new = scaled_O_prev + tl.dot(P, v_tile)

        m = m_new
        l = l_new
        O = O_new
    
    tl.store(O_block_ptr, O / l [:, None])
    # tl.store(l_block_ptr, m + tl.log(l))
    tl.store(l_ptrs, m + tl.log(l), mask=l_mask)
     
class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V: torch.tensor, is_causal=False):
        if K.shape != V.shape:
            raise "K and V shape should match."
        
        B, S, D = Q.shape
        O = torch.empty(B, S, D, device=Q.device)
        L = torch.empty(B, S, device=Q.device)
        scale = 1.0/math.sqrt(D)

        Nq = S   
        Nk = S
        d  = D

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        Tq = math.ceil(Nq/ctx.Q_TILE_SIZE)
        Tk = math.ceil(Nk/ctx.K_TILE_SIZE)

        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()

        flash_attention_forward_kernel[(Tq, B)](Q, K, V, O, L,
        stride_qb = stride_qb,
        stride_qq = stride_qq,
        stride_qd = stride_qd,
        stride_kb = stride_kb,
        stride_kk = stride_kk,
        stride_kd = stride_kd,
        stride_vb = stride_vb,
        stride_vk = stride_vk,
        stride_vd = stride_vd,
        stride_ob = stride_ob,
        stride_oq = stride_oq,
        stride_od = stride_od,
        stride_lb = stride_lb,
        stride_lq = stride_lq,
        N_QUERIES=Nq,
        N_KEYS=Nk,
        scale=scale,
        D=D,
        Q_TILE_SIZE=ctx.Q_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE,  IS_CAUSAL=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
