import torch
from einops import einsum
from cs336_basics import softmax
import math


def scaled_dot_product_attention(Q, K, V, mask):
    """
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided).
    The implementation should return an output with the shape (batch_size,..., d_v).
    """
    score = einsum(Q, K, "... n dk,... m dk ->... n m") / math.sqrt(Q.shape[-1])
    while mask.ndim < score.ndim:
        mask = mask.unsqueeze(0)  # keeps stacking 1s on the left
    # ensure boolean mask on same device
    mask = mask.to(dtype=torch.bool, device=score.device)
    # in-place masking, no extra atten_mask
    score.masked_fill_(~mask, float("-inf"))
    score = softmax.softmax(score, -1)
    attention = einsum(score, V, "... n m,... m dv -> ... n dv")
    return attention
