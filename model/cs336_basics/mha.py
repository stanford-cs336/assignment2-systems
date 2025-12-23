import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from einops import rearrange, einsum
import math

from .scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.RoPE import RoPE


class MHA(nn.Module):
    def __init__(self, d_model, num_heads, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # learable parameter
        self.dk = int(self.d_model / num_heads)
        self.dv = int(self.d_model / num_heads)
        # define weights W_q W_k W_v and W_o
        sigma = math.sqrt(2 / (self.dk + self.d_model))
        self.Wq = nn.Parameter(
            torch.empty(size=(num_heads * self.dk, self.d_model), device=device)
        )
        init.trunc_normal_(self.Wq, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.Wk = nn.Parameter(
            torch.empty(size=(num_heads * self.dk, self.d_model), device=device)
        )
        init.trunc_normal_(self.Wk, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

        sigma = math.sqrt(2 / (self.dv + self.d_model))

        self.Wv = nn.Parameter(
            torch.empty(size=(num_heads * self.dv, self.d_model), device=device)
        )
        init.trunc_normal_(self.Wv, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.Wo = nn.Parameter(
            torch.empty(size=(self.d_model, num_heads * self.dv), device=device)
        )
        init.trunc_normal_(self.Wo, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def mha_self_attention(
        self, x, theta=None, token_position=None, max_seq_len=None, device=None
    ):

        if theta != None:
            R = RoPE(
                theta, int(x.shape[-1] / self.num_heads), max_seq_len, device=device
            )
        # compute Q,K,V
        Q = einsum(self.Wq, x, "i j,... t j -> ... t i")
        Q = rearrange(Q, "... b t (h d) -> ... b h t d", h=self.num_heads)
        if theta != None:
            Q = R.forward(Q, token_position)

        K = einsum(self.Wk, x, "i j,... t j ->... t i")
        K = rearrange(K, "... b t (h d) -> ... b h t d", h=self.num_heads)
        if theta != None:
            K = R.forward(K, token_position)

        V = einsum(self.Wv, x, "i j, ... t j ->... t i")
        V = rearrange(V, "... b t (h d) -> ... b h t d", h=self.num_heads)

        M = torch.tril(
            torch.ones(
                x.shape[0],
                self.num_heads,
                x.shape[-2],
                x.shape[-2],
                dtype=torch.bool,
                device=device,
            ),
        )
        output_mha = scaled_dot_product_attention(Q, K, V, mask=M)
        output_mha = rearrange(output_mha, "... b h t d -> ... b t (h d)")
        y = einsum(self.Wo, output_mha, "i j,... n j -> ... n i")
        return y
