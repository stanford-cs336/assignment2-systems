from __future__ import annotations

import math
import os

try:
    import triton  # pyright: ignore[reportMissingImports]
    import triton.language as tl  # pyright: ignore[reportMissingImports]
except ImportError:
    triton = None  # type: ignore[misc, assignment]
    tl = None  # type: ignore[misc, assignment]

import torch
from einops import rearrange
from torch.autograd import Function

Q_TILE_SIZE: int = 32
K_TILE_SIZE: int = 32


def flash_attention2_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    q_tile_size: int = Q_TILE_SIZE,
    k_tile_size: int = K_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_queries, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros_like(q)
    log_sum_exp = torch.empty(batch_size, num_queries, device=q.device, dtype=torch.float32)

    query_tiles = rearrange(q, "b (t_q bq) d -> t_q b bq d", bq=q_tile_size)
    key_tiles = rearrange(k, "b (t_k bk) d -> t_k b bk d", bk=k_tile_size)
    value_tiles = rearrange(v, "b (t_v bv) d -> t_v b bv d", bv=k_tile_size)

    num_query_tiles = num_queries // q_tile_size
    num_key_tiles = k.shape[1] // k_tile_size

    for query_tile_index in range(num_query_tiles):
        query_tile = query_tiles[query_tile_index]
        output_accumulator = torch.zeros(batch_size, q_tile_size, head_dim, device=q.device, dtype=torch.float32)
        log_normalizer = torch.zeros(batch_size, q_tile_size, device=q.device, dtype=torch.float32)
        running_max = torch.full((batch_size, q_tile_size), float("-inf"), device=q.device, dtype=torch.float32)

        for key_tile_index in range(num_key_tiles):
            key_tile = key_tiles[key_tile_index]
            value_tile = value_tiles[key_tile_index]
            attention_scores = torch.einsum("b i d, b j d -> b i j", query_tile.float(), key_tile.float()) * scale
            if is_causal:
                query_positions = torch.arange(
                    query_tile_index * q_tile_size, (query_tile_index + 1) * q_tile_size, device=q.device
                )
                key_positions = torch.arange(
                    key_tile_index * k_tile_size, (key_tile_index + 1) * k_tile_size, device=q.device
                )
                causal_mask = query_positions[:, None] >= key_positions[None, :]
                attention_scores = attention_scores.masked_fill(~causal_mask, -1e6)

            new_max = torch.maximum(running_max, attention_scores.amax(dim=-1))
            unnormalized_attention = torch.exp(attention_scores - new_max.unsqueeze(-1))
            new_log_normalizer = torch.exp(running_max - new_max) * log_normalizer + unnormalized_attention.sum(dim=-1)
            output_accumulator = (
                torch.exp((running_max - new_max).unsqueeze(-1)) * output_accumulator
                + torch.einsum("bij,bjd->bid", unnormalized_attention, value_tile.float())
            )
            running_max, log_normalizer = new_max, new_log_normalizer

        output_tile = (output_accumulator / log_normalizer.unsqueeze(-1)).to(q.dtype)
        log_sum_exp_tile = running_max + torch.log(log_normalizer)
        output[:, query_tile_index * q_tile_size : (query_tile_index + 1) * q_tile_size] = output_tile
        log_sum_exp[:, query_tile_index * q_tile_size : (query_tile_index + 1) * q_tile_size] = log_sum_exp_tile

    return output, log_sum_exp


if triton is not None:
    _tl = tl

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        D: _tl.constexpr,
        Q_TILE_SIZE: _tl.constexpr,
        K_TILE_SIZE: _tl.constexpr,
        is_causal: _tl.constexpr,
    ):
        query_tile_index = _tl.program_id(0)
        batch_index = _tl.program_id(1)

        Q_block_ptr = _tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = _tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = _tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        K_block_ptr = _tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = _tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        q = _tl.load(Q_block_ptr)
        q_f = q.to(_tl.float32)

        o_i = _tl.zeros((Q_TILE_SIZE, D), dtype=_tl.float32)
        l_acc = _tl.zeros((Q_TILE_SIZE,), dtype=_tl.float32)
        m_acc = _tl.full((Q_TILE_SIZE,), float("-inf"), dtype=_tl.float32)

        for kv_i in range(_tl.cdiv(N_KEYS, K_TILE_SIZE)):
            tk = _tl.load(K_block_ptr)
            tv = _tl.load(V_block_ptr)
            tk_f = tk.to(_tl.float32)

            s = _tl.dot(q_f, _tl.trans(tk_f)) * scale

            if is_causal:
                q_idx = query_tile_index * Q_TILE_SIZE + _tl.arange(0, Q_TILE_SIZE)
                k_idx = kv_i * K_TILE_SIZE + _tl.arange(0, K_TILE_SIZE)
                causal_ok = q_idx[:, None] >= k_idx[None, :]
                s = _tl.where(causal_ok, s, -1e6)

            m_new = _tl.maximum(m_acc, _tl.max(s, 1))
            p_tilde = _tl.exp(s - m_new[:, None])
            l_new = _tl.exp(m_acc - m_new) * l_acc + _tl.sum(p_tilde, 1)
            p_v = _tl.dot(p_tilde.to(tv.dtype), tv).to(_tl.float32)
            o_i = _tl.exp(m_acc - m_new)[:, None] * o_i + p_v
            m_acc, l_acc = m_new, l_new

            K_block_ptr = _tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = _tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        o_tile = o_i / l_acc[:, None]
        l_tile = m_acc + _tl.log(l_acc)
        _tl.store(O_block_ptr, o_tile.to(O_block_ptr.type.element_ty))
        _tl.store(L_block_ptr, l_tile)

    @triton.jit
    def flash_bwd_dkdv(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        Delta_ptr,
        dK_ptr,
        dV_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_lb,
        stride_lq,
        stride_delb,
        stride_delq,
        stride_dkb,
        stride_dkk,
        stride_dkd,
        stride_dvb,
        stride_dvk,
        stride_dvd,
        N_QUERIES,
        N_KEYS,
        scale,
        D_HEAD: _tl.constexpr,
        Q_TILE: _tl.constexpr,
        K_TILE: _tl.constexpr,
        is_causal: _tl.constexpr,
    ):
        """One outer key tile × inner query tiles: accumulate ``dK``, ``dV`` (Algorithm 2, first nested loop)."""
        key_tile_idx = _tl.program_id(0)
        batch_index = _tl.program_id(1)
        k0 = key_tile_idx * K_TILE

        K_block_ptr = _tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D_HEAD),
            strides=(stride_kk, stride_kd),
            offsets=(k0, 0),
            block_shape=(K_TILE, D_HEAD),
            order=(1, 0),
        )
        V_block_ptr = _tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D_HEAD),
            strides=(stride_vk, stride_vd),
            offsets=(k0, 0),
            block_shape=(K_TILE, D_HEAD),
            order=(1, 0),
        )
        k_tile = _tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)
        v_tile = _tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)

        dk_acc = _tl.zeros((K_TILE, D_HEAD), dtype=_tl.float32)
        dv_acc = _tl.zeros((K_TILE, D_HEAD), dtype=_tl.float32)
        num_q_tiles = _tl.cdiv(N_QUERIES, Q_TILE)

        for qi in range(num_q_tiles):
            q0 = qi * Q_TILE
            Q_block_ptr = _tl.make_block_ptr(
                Q_ptr + batch_index * stride_qb,
                shape=(N_QUERIES, D_HEAD),
                strides=(stride_qq, stride_qd),
                offsets=(q0, 0),
                block_shape=(Q_TILE, D_HEAD),
                order=(1, 0),
            )
            dO_block_ptr = _tl.make_block_ptr(
                dO_ptr + batch_index * stride_dob,
                shape=(N_QUERIES, D_HEAD),
                strides=(stride_doq, stride_dod),
                offsets=(q0, 0),
                block_shape=(Q_TILE, D_HEAD),
                order=(1, 0),
            )
            L_subptr = _tl.make_block_ptr(
                L_ptr + batch_index * stride_lb,
                shape=(N_QUERIES,),
                strides=(stride_lq,),
                offsets=(q0,),
                block_shape=(Q_TILE,),
                order=(0,),
            )
            Del_subptr = _tl.make_block_ptr(
                Delta_ptr + batch_index * stride_delb,
                shape=(N_QUERIES,),
                strides=(stride_delq,),
                offsets=(q0,),
                block_shape=(Q_TILE,),
                order=(0,),
            )
            q_tile_q = _tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)
            do_tile = _tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)
            l_row = _tl.load(L_subptr, boundary_check=(0,), padding_option="zero").to(_tl.float32)
            del_row = _tl.load(Del_subptr, boundary_check=(0,), padding_option="zero").to(_tl.float32)

            s = _tl.dot(q_tile_q, _tl.trans(k_tile)) * scale
            if is_causal:
                offs_q = q0 + _tl.arange(0, Q_TILE)
                offs_k = k0 + _tl.arange(0, K_TILE)
                causal_ok = offs_q[:, None] >= offs_k[None, :]
                s = _tl.where(causal_ok, s, -1e6)
            p = _tl.exp(s - l_row[:, None])

            dv_acc += _tl.dot(_tl.trans(p), do_tile)
            dp = _tl.dot(do_tile, _tl.trans(v_tile))
            ds = p * (dp - del_row[:, None])
            if is_causal:
                offs_q = q0 + _tl.arange(0, Q_TILE)
                offs_k = k0 + _tl.arange(0, K_TILE)
                causal_ok = offs_q[:, None] >= offs_k[None, :]
                ds = _tl.where(causal_ok, ds, 0.0)
            dk_acc += _tl.dot(_tl.trans(ds), q_tile_q) * scale

        dK_block_ptr = _tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb,
            shape=(N_KEYS, D_HEAD),
            strides=(stride_dkk, stride_dkd),
            offsets=(k0, 0),
            block_shape=(K_TILE, D_HEAD),
            order=(1, 0),
        )
        dV_block_ptr = _tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb,
            shape=(N_KEYS, D_HEAD),
            strides=(stride_dvk, stride_dvd),
            offsets=(k0, 0),
            block_shape=(K_TILE, D_HEAD),
            order=(1, 0),
        )
        _tl.store(dK_block_ptr, dk_acc.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
        _tl.store(dV_block_ptr, dv_acc.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

    @triton.jit
    def flash_bwd_dq(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        Delta_ptr,
        dQ_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_lb,
        stride_lq,
        stride_delb,
        stride_delq,
        stride_dqb,
        stride_dqq,
        stride_dqd,
        N_QUERIES,
        N_KEYS,
        scale,
        D_HEAD: _tl.constexpr,
        Q_TILE: _tl.constexpr,
        K_TILE: _tl.constexpr,
        is_causal: _tl.constexpr,
    ):
        """One query tile × inner key tiles: accumulate ``dQ`` (Algorithm 2, second nested loop)."""
        query_tile_idx = _tl.program_id(0)
        batch_index = _tl.program_id(1)
        q0 = query_tile_idx * Q_TILE

        Q_block_ptr = _tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D_HEAD),
            strides=(stride_qq, stride_qd),
            offsets=(q0, 0),
            block_shape=(Q_TILE, D_HEAD),
            order=(1, 0),
        )
        dO_block_ptr = _tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D_HEAD),
            strides=(stride_doq, stride_dod),
            offsets=(q0, 0),
            block_shape=(Q_TILE, D_HEAD),
            order=(1, 0),
        )
        L_subptr = _tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(q0,),
            block_shape=(Q_TILE,),
            order=(0,),
        )
        Del_subptr = _tl.make_block_ptr(
            Delta_ptr + batch_index * stride_delb,
            shape=(N_QUERIES,),
            strides=(stride_delq,),
            offsets=(q0,),
            block_shape=(Q_TILE,),
            order=(0,),
        )
        q_tile_q = _tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)
        do_tile = _tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)
        l_row = _tl.load(L_subptr, boundary_check=(0,), padding_option="zero").to(_tl.float32)
        del_row = _tl.load(Del_subptr, boundary_check=(0,), padding_option="zero").to(_tl.float32)

        dq_acc = _tl.zeros((Q_TILE, D_HEAD), dtype=_tl.float32)
        num_k_tiles = _tl.cdiv(N_KEYS, K_TILE)

        for kj in range(num_k_tiles):
            k0 = kj * K_TILE
            K_block_ptr = _tl.make_block_ptr(
                K_ptr + batch_index * stride_kb,
                shape=(N_KEYS, D_HEAD),
                strides=(stride_kk, stride_kd),
                offsets=(k0, 0),
                block_shape=(K_TILE, D_HEAD),
                order=(1, 0),
            )
            V_block_ptr = _tl.make_block_ptr(
                V_ptr + batch_index * stride_vb,
                shape=(N_KEYS, D_HEAD),
                strides=(stride_vk, stride_vd),
                offsets=(k0, 0),
                block_shape=(K_TILE, D_HEAD),
                order=(1, 0),
            )
            k_tile = _tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)
            v_tile = _tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(_tl.float32)

            s = _tl.dot(q_tile_q, _tl.trans(k_tile)) * scale
            if is_causal:
                offs_q = q0 + _tl.arange(0, Q_TILE)
                offs_k = k0 + _tl.arange(0, K_TILE)
                causal_ok = offs_q[:, None] >= offs_k[None, :]
                s = _tl.where(causal_ok, s, -1e6)
            p = _tl.exp(s - l_row[:, None])
            dp = _tl.dot(do_tile, _tl.trans(v_tile))
            ds = p * (dp - del_row[:, None])
            if is_causal:
                offs_q = q0 + _tl.arange(0, Q_TILE)
                offs_k = k0 + _tl.arange(0, K_TILE)
                causal_ok = offs_q[:, None] >= offs_k[None, :]
                ds = _tl.where(causal_ok, ds, 0.0)
            dq_acc += _tl.dot(ds, k_tile) * scale

        dQ_block_ptr = _tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D_HEAD),
            strides=(stride_dqq, stride_dqd),
            offsets=(q0, 0),
            block_shape=(Q_TILE, D_HEAD),
            order=(1, 0),
        )
        _tl.store(dQ_block_ptr, dq_acc.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

else:
    flash_fwd_kernel = None  # type: ignore[misc, assignment]
    flash_bwd_dkdv = None  # type: ignore[misc, assignment]
    flash_bwd_dq = None  # type: ignore[misc, assignment]


def launch_flash_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    scale: float,
    is_causal: bool,
    q_tile: int = Q_TILE_SIZE,
    k_tile: int = K_TILE_SIZE,
) -> None:
    batch_size, num_queries, head_dim = q.shape
    num_query_tiles = (num_queries + q_tile - 1) // q_tile
    grid = (num_query_tiles, batch_size)

    flash_fwd_kernel[grid](  # type: ignore[index]
        q,
        k,
        v,
        output,
        log_sum_exp,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        log_sum_exp.stride(0),
        log_sum_exp.stride(1),
        num_queries,
        k.shape[1],
        scale,
        head_dim,
        q_tile,
        k_tile,
        is_causal,
    )


def _triton_bwd_qk_tiles(nq: int, nk: int) -> tuple[int, int]:
    """Pick ``Q_TILE`` / ``K_TILE`` for backward kernels.

    Larger tiles cut inner-loop count but increase per-CTA **shared** memory (``Q_TILE×K_TILE``
    softmax scratch in fp32, etc.). On B200 we have seen ``OutOfResources`` with 256² blocks
    (~1328 KiB requested vs ~232 KiB limit).  **64** is a conservative default; raise only via
    ``FLASH_ATTENTION_BWD_TILE`` when you have verified the kernel fits.
    """
    raw_env = os.environ.get("FLASH_ATTENTION_BWD_TILE", "").strip()
    if raw_env:
        t = max(16, int(raw_env))
        return min(t, nq), min(t, nk)
    if nq >= 8192 or nk >= 8192:
        t = 64
    elif nq >= 512 or nk >= 512:
        t = 32
    else:
        t = 32
    return max(16, min(t, nq)), max(16, min(t, nk))


def launch_flash_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    scale: float,
    is_causal: bool,
    q_tile: int | None = None,
    k_tile: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiled FA-2 backward (§4.2.3): two kernels, no full ``(T×T)`` attention matrix."""
    delta = (output.to(torch.float32) * grad_output.to(torch.float32)).sum(dim=-1)
    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    batch_size, nq, head_dim = q.shape
    nk = k.shape[1]
    if q_tile is None or k_tile is None:
        q_tile, k_tile = _triton_bwd_qk_tiles(nq, nk)
    num_q_tiles = triton.cdiv(nq, q_tile)  # type: ignore[arg-type]
    num_k_tiles = triton.cdiv(nk, k_tile)  # type: ignore[arg-type]

    try:
        flash_bwd_dkdv[(num_k_tiles, batch_size)](  # type: ignore[index]
            q,
            k,
            v,
            grad_output,
            log_sum_exp,
            delta,
            grad_k,
            grad_v,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            log_sum_exp.stride(0),
            log_sum_exp.stride(1),
            delta.stride(0),
            delta.stride(1),
            grad_k.stride(0),
            grad_k.stride(1),
            grad_k.stride(2),
            grad_v.stride(0),
            grad_v.stride(1),
            grad_v.stride(2),
            nq,
            nk,
            scale,
            head_dim,
            q_tile,
            k_tile,
            is_causal,
        )
        flash_bwd_dq[(num_q_tiles, batch_size)](  # type: ignore[index]
            q,
            k,
            v,
            grad_output,
            log_sum_exp,
            delta,
            grad_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            log_sum_exp.stride(0),
            log_sum_exp.stride(1),
            delta.stride(0),
            delta.stride(1),
            grad_q.stride(0),
            grad_q.stride(1),
            grad_q.stride(2),
            nq,
            nk,
            scale,
            head_dim,
            q_tile,
            k_tile,
            is_causal,
        )
    except Exception as exc:
        if triton is not None:
            from triton.runtime.errors import OutOfResources

            if isinstance(exc, OutOfResources):
                import warnings

                warnings.warn(
                    f"Triton FA backward exceeded GPU resources ({exc!s}); using Python tiled backward.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return flash_attention_backward_tiled(
                    q, k, v, output, grad_output, log_sum_exp, is_causal=is_causal
                )
        raise

    return grad_q, grad_k, grad_v


def flash_attention_backward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense backward (O(T²) memory); for long context use :func:`flash_attention_backward_tiled`."""
    original_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    output = output.to(torch.float32)
    grad_output = grad_output.to(torch.float32)
    sqrt_head_dim = math.sqrt(q.shape[-1])
    diagonal_correction = torch.einsum("bqd,bqd->bq", output, grad_output)
    attention_scores = torch.einsum("bqd,bkd->bqk", q, k) / sqrt_head_dim
    if is_causal:
        query_positions = torch.arange(q.shape[1], device=q.device)
        key_positions = torch.arange(k.shape[1], device=q.device)
        attention_scores = attention_scores.masked_fill(query_positions[:, None] < key_positions[None, :], -1e6)
    attention_weights = torch.exp(attention_scores - log_sum_exp.unsqueeze(-1))
    grad_v = torch.einsum("bqk,bqd->bkd", attention_weights, grad_output)
    grad_attention_weights = torch.einsum("bqd,bkd->bqk", grad_output, v)
    grad_scores = attention_weights * (grad_attention_weights - diagonal_correction.unsqueeze(-1))
    if is_causal:
        grad_scores = grad_scores.masked_fill(query_positions[:, None] < key_positions[None, :], 0.0)
    grad_q = torch.einsum("bqk,bkd->bqd", grad_scores, k) / sqrt_head_dim
    grad_k = torch.einsum("bqk,bqd->bkd", grad_scores, q) / sqrt_head_dim
    return grad_q.to(original_dtype), grad_k.to(original_dtype), grad_v.to(original_dtype)


def flash_attention_backward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    is_causal: bool,
    q_tile_size: int = Q_TILE_SIZE,
    k_tile_size: int = K_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same math as :func:`flash_attention_backward_pytorch` but only materializes ``q_tile×k_tile`` score blocks."""
    original_dtype = q.dtype
    nq, d_hat = q.shape[1], q.shape[2]
    nk = k.shape[1]
    # Python double loop cost is O((T / tile)²) per layer. At T=32k, tile=512 ⇒ ~2k pairs/layer ⇒ tens of
    # thousands of small launches across the network — use much larger tiles on long sequences only.
    if nq >= 32_768:
        q_tile_size = k_tile_size = 4096
    elif nq >= 16_384:
        q_tile_size = k_tile_size = 2048
    elif nq >= 4096:
        q_tile_size = k_tile_size = 1024
    elif nq >= 1024:
        q_tile_size = k_tile_size = 256
    # else: keep caller defaults (32) for short sequences / tests
    q_tile_size = max(1, min(nq, q_tile_size))
    k_tile_size = max(1, min(nk, k_tile_size))
    scale = 1.0 / math.sqrt(d_hat)

    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    output_f = output.to(torch.float32)
    go_f = grad_output.to(torch.float32)
    lse = log_sum_exp.to(torch.float32)

    d_vec = (output_f * go_f).sum(dim=-1)
    num_q_tiles = (nq + q_tile_size - 1) // q_tile_size
    num_k_tiles = (nk + k_tile_size - 1) // k_tile_size

    for qi in range(num_q_tiles):
        q_lo = qi * q_tile_size
        q_hi = min(q_lo + q_tile_size, nq)
        q_blk = q_f[:, q_lo:q_hi, :]
        go_blk = go_f[:, q_lo:q_hi, :]
        d_blk = d_vec[:, q_lo:q_hi]
        lse_blk = lse[:, q_lo:q_hi]

        for ki in range(num_k_tiles):
            k_lo = ki * k_tile_size
            k_hi = min(k_lo + k_tile_size, nk)

            if is_causal and q_hi - 1 < k_lo:
                continue

            k_blk = k_f[:, k_lo:k_hi, :]
            v_blk = v_f[:, k_lo:k_hi, :]

            s = torch.einsum("bqd,bkd->bqk", q_blk, k_blk) * scale
            if is_causal:
                q_idx = torch.arange(q_lo, q_hi, device=q.device).unsqueeze(1)
                k_idx = torch.arange(k_lo, k_hi, device=q.device).unsqueeze(0)
                causal_ok = q_idx >= k_idx
                s = s.masked_fill(~causal_ok, -1e6)

            p = torch.exp(s - lse_blk.unsqueeze(-1))
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

            grad_v[:, k_lo:k_hi, :] += torch.einsum("bqk,bqd->bkd", p, go_blk)

            gp = torch.einsum("bqd,bkd->bqk", go_blk, v_blk)
            ds = p * (gp - d_blk.unsqueeze(-1))
            if is_causal:
                ds = ds.masked_fill(~causal_ok, 0.0)

            grad_q[:, q_lo:q_hi, :] += torch.einsum("bqk,bkd->bqd", ds, k_blk) * scale
            grad_k[:, k_lo:k_hi, :] += torch.einsum("bqk,bqd->bkd", ds, q_blk) * scale

    return grad_q.to(original_dtype), grad_k.to(original_dtype), grad_v.to(original_dtype)


# §4.2.3: ``torch.compile`` on the **dense** reference — fine for short ``T`` (tests); at large ``T``
# Inductor materializes ``(B, T, T)`` and OOMs.
#
# For ``T`` above the cutoff the **default** is eager :func:`flash_attention_backward_tiled` (large
# tiles + cuBLAS ``einsum``) — fastest on §9-scale models we've measured.  Optional §4.2.3 two-kernel
# Triton backward: set ``FLASH_ATTENTION_BACKWARD_TRITON=1`` (tune ``FLASH_ATTENTION_BWD_TILE`` if
# you hit shared-memory limits).  ``torch.compile(flash_attention_backward_tiled)`` stays opt-in via
# ``FLASH_ATTENTION_BACKWARD_COMPILE_TILED=1`` (Inductor may OOM).
_flash_attention_backward_dense_compiled = torch.compile(flash_attention_backward_pytorch)
_flash_attention_backward_chunked_compiled = torch.compile(
    flash_attention_backward_tiled,
    fullgraph=False,
)


def _flash_attention_backward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nq = q.shape[1]
    if os.environ.get("FLASH_ATTENTION_BACKWARD_EAGER_TILED", "").strip().lower() in ("1", "true", "yes"):
        return flash_attention_backward_tiled(
            q, k, v, output, grad_output, log_sum_exp, is_causal=is_causal
        )
    compile_max = int(os.environ.get("FLASH_ATTENTION_BACKWARD_COMPILE_MAX_SEQ", "4096"))
    # Force dense compiled path (will OOM at §9 ``T``; for debugging / tiny runs only).
    if os.environ.get("FLASH_ATTENTION_BACKWARD_COMPILED_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return _flash_attention_backward_dense_compiled(
            q, k, v, output, grad_output, log_sum_exp, is_causal=is_causal
        )
    if nq <= compile_max:
        return _flash_attention_backward_dense_compiled(
            q, k, v, output, grad_output, log_sum_exp, is_causal=is_causal
        )
    if os.environ.get("FLASH_ATTENTION_BACKWARD_COMPILE_TILED", "").strip().lower() in ("1", "true", "yes"):
        return _flash_attention_backward_chunked_compiled(
            q, k, v, output, grad_output, log_sum_exp, is_causal=is_causal
        )
    _use_triton_bwd = (
        flash_bwd_dkdv is not None
        and q.is_cuda
        and os.environ.get("FLASH_ATTENTION_BACKWARD_TRITON", "").strip().lower() in ("1", "true", "yes")
    )
    if _use_triton_bwd:
        return launch_flash_backward_triton(
            q,
            k,
            v,
            output,
            grad_output,
            log_sum_exp,
            scale=1.0 / math.sqrt(q.shape[-1]),
            is_causal=is_causal,
        )
    return flash_attention_backward_tiled(
        q, k, v, output, grad_output, log_sum_exp, is_causal=is_causal
    )


class FlashAttention2PyTorch(Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):  # noqa: FBT001, FBT002
        output, log_sum_exp = flash_attention2_forward_pytorch(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, output, log_sum_exp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        q, k, v, output, log_sum_exp = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_attention_backward_dispatch(
            q,
            k,
            v,
            output,
            grad_output,
            log_sum_exp,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


class FlashAttention2Triton(FlashAttention2PyTorch):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        batch_size, num_queries, head_dim = q.shape
        output = torch.empty_like(q)
        log_sum_exp = torch.empty(batch_size, num_queries, device=q.device, dtype=torch.float32)
        launch_flash_forward_triton(q, k, v, output, log_sum_exp, scale=1.0 / math.sqrt(head_dim), is_causal=is_causal)
        ctx.save_for_backward(q, k, v, output, log_sum_exp)
        ctx.is_causal = is_causal
        return output
