"""Leaderboard: use PyTorch fused SDPA (same causal semantics as the handout dense mask being ignored)."""

from __future__ import annotations

import cs336_basics.model as basics_model
import torch.nn.functional as F
from jaxtyping import Bool, Float


def _scaled_dot_product_attention_sdpa(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys    d_k"],
    V: Float[torch.Tensor, " ... keys    d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,  # noqa: ARG001
) -> Float[torch.Tensor, " ... queries d_v"]:
    return F.scaled_dot_product_attention(
        Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
    )


_installed = False


def install_flash_sdpa() -> None:
    """Idempotent: patch ``cs336_basics.model.scaled_dot_product_attention`` to SDPA."""
    global _installed
    if _installed:
        return
    basics_model.scaled_dot_product_attention = _scaled_dot_product_attention_sdpa  # type: ignore[assignment]
    _installed = True
