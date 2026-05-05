from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


def _vocab_row_slices(vocab_size: int, rows_per_chunk: int) -> Iterator[tuple[int, int]]:
    for vocab_row_begin in range(0, vocab_size, rows_per_chunk):
        vocab_row_end = min(vocab_row_begin + rows_per_chunk, vocab_size)
        yield vocab_row_begin, vocab_row_end


class ChunkedVocabCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_token_ids: torch.Tensor,
        vocab_chunk_rows: int,
    ) -> torch.Tensor:
        if vocab_chunk_rows < 1:
            raise ValueError("vocab_chunk_rows must be positive")

        num_tokens, model_dim = hidden_states.shape
        vocab_size, weight_in_dim = lm_head_weight.shape
        if weight_in_dim != model_dim:
            raise ValueError(f"lm_head_weight has d_in={weight_in_dim}, hidden has d_model={model_dim}")
        if target_token_ids.shape != (num_tokens,):
            raise ValueError("target_token_ids must be 1D with one target per token row")

        max_logit = torch.full((num_tokens,), float("-inf"), device=hidden_states.device, dtype=torch.float32)
        sum_exp = torch.zeros(num_tokens, device=hidden_states.device, dtype=torch.float32)
        logit_at_target = torch.zeros(num_tokens, device=hidden_states.device, dtype=torch.float32)

        hidden_fp32 = hidden_states.float()
        for row_begin, row_end in _vocab_row_slices(vocab_size, vocab_chunk_rows):
            weight_chunk = lm_head_weight[row_begin:row_end].float()
            logits_chunk = (hidden_fp32 @ weight_chunk.T).float()

            chunk_max_per_row = logits_chunk.max(dim=-1).values
            updated_max = torch.maximum(max_logit, chunk_max_per_row)
            rescale = torch.exp(max_logit - updated_max)
            sum_exp = sum_exp * rescale + torch.exp(logits_chunk - updated_max.unsqueeze(-1)).sum(dim=-1)
            max_logit = updated_max

            target_in_chunk = (target_token_ids >= row_begin) & (target_token_ids < row_end)
            if target_in_chunk.any():
                row_ix = target_in_chunk.nonzero(as_tuple=True)[0]
                col_in_chunk = target_token_ids[row_ix] - row_begin
                logit_at_target[row_ix] = logits_chunk[row_ix, col_in_chunk]

        log_sum_exp = max_logit + torch.log(sum_exp.clamp(min=torch.finfo(torch.float32).tiny))
        loss = (log_sum_exp - logit_at_target).sum()

        ctx.save_for_backward(hidden_states, lm_head_weight, target_token_ids)
        ctx.log_sum_exp = log_sum_exp
        ctx.vocab_chunk_rows = vocab_chunk_rows
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        hidden_states, lm_head_weight, target_token_ids = ctx.saved_tensors
        log_sum_exp: torch.Tensor = ctx.log_sum_exp
        vocab_chunk_rows: int = ctx.vocab_chunk_rows
        num_tokens, model_dim = hidden_states.shape
        vocab_size, _ = lm_head_weight.shape

        loss_scale = float(grad_output)
        grad_hidden = torch.zeros(num_tokens, model_dim, device=hidden_states.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(lm_head_weight, dtype=torch.float32)

        hidden_fp32 = hidden_states.float()
        for row_begin, row_end in _vocab_row_slices(vocab_size, vocab_chunk_rows):
            weight_chunk = lm_head_weight[row_begin:row_end].float()
            logits_chunk = (hidden_fp32 @ weight_chunk.T).float()

            probs = torch.exp(logits_chunk - log_sum_exp.unsqueeze(-1))
            grad_logits = probs * loss_scale

            target_in_chunk = (target_token_ids >= row_begin) & (target_token_ids < row_end)
            if target_in_chunk.any():
                row_ix = target_in_chunk.nonzero(as_tuple=True)[0]
                col_in_chunk = target_token_ids[row_ix] - row_begin
                grad_logits[row_ix, col_in_chunk] -= loss_scale

            grad_hidden.addmm_(grad_logits, weight_chunk)
            grad_weight[row_begin:row_end] = grad_logits.T @ hidden_fp32

        grad_hidden = grad_hidden.to(dtype=hidden_states.dtype)
        grad_weight = grad_weight.to(dtype=lm_head_weight.dtype)
        return grad_hidden, grad_weight, None, None


def chunked_lm_head_cross_entropy_loss(
    hidden_states: torch.Tensor,
    gathered_lm_head_weight: torch.Tensor,
    target_token_ids: torch.Tensor,
    vocab_chunk_rows: int,
) -> torch.Tensor:
    batch, seq_len, _ = hidden_states.shape
    flat_hidden = hidden_states.reshape(batch * seq_len, -1)
    flat_targets = target_token_ids.reshape(-1).long()
    return ChunkedVocabCrossEntropy.apply(flat_hidden, gathered_lm_head_weight, flat_targets, vocab_chunk_rows)
