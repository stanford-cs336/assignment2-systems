import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math
import numpy as np

from einops import einsum
from .Linear import Linear


class RoPE:
    """
    To inject positional information into the model, we will implement Rotary Position Embeddings [Su et al.,
    2021], often called RoPE. For a given query token q(i) = Wqx(i),is d dim, at token position i, we will apply a
    pairwise rotation matrix Ri, giving us q'(i) = R(i) q(i) = R(i) * Wq * x(i). Here, R(i) will rotate pairs of embedding
    elements q(i){2k-1:2k} as 2d vectors by the angle theta_i,k = i / Θ^{(2k-1)/d} for k ∈ {1, . . . , d/2} and some constant Θ.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # Generate R_i = [cos -sin;sin cos]
        theta_list = []
        for i in range(self.max_seq_len):
            theta_list.append(
                [
                    i / self.theta ** ((2 * k) / self.d_k)
                    for k in range(0, int(self.d_k * 0.5))
                ]
            )

        # construct R_i's from theta_list
        self.R = torch.zeros(size=(self.max_seq_len, self.d_k, self.d_k), device=device)
        for m in range(len(theta_list)):
            for k in range(int(0.5 * self.d_k)):
                R_i = torch.empty(2, 2)
                R_i[0] = torch.tensor(
                    [math.cos(theta_list[m][k]), -1 * math.sin(theta_list[m][k])]
                )
                R_i[1] = torch.tensor(
                    [1 * math.sin(theta_list[m][k]), math.cos(theta_list[m][k])]
                )
                self.R[m, 2 * k, 2 * k : 2 * k + 2] = R_i[0]
                self.R[m, 2 * k + 1, 2 * k : 2 * k + 2] = R_i[1]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        """
        # construct R_i's from theta_list
        R_sel = self.R[token_positions]  # [T, D, D]
        y = einsum(R_sel, x, "... t e d, ... b h t d -> ... b h t e")
        return y
