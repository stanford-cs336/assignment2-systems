import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from einops import rearrange, einsum
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, W=None, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_size = in_features
        self.out_size = out_features
        self.device = device
        self.dtype = dtype
        if W != None:
            self.W = W
        else:
            sigma = math.sqrt(2 / (in_features + out_features))
            self.W = nn.Parameter(torch.empty(out_features, in_features, device=device))
            init.trunc_normal_(self.W, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the
        input. y=Wx
        """
        y = einsum(self.W, x, "out ip , batch time ip -> batch time out")
        return y
