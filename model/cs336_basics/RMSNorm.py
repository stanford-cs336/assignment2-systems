import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from einops import rearrange, einsum


class RMSNorm(nn.Module):
    """
    1. Prevents vanishing and exploding gradient problems.
    2. Looks **across the batch**.
    3. Each **feature/column** is normalized using statistics from all samples in the batch.
    4. Normalization depends on batch size & composition.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args :
            d_model : Dim of the model
            eps : epsilon
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g_i = nn.Parameter(
            torch.ones(size=(1, d_model), dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape  and
        return a tensor of the same shape.
        Args :
            x : input of shape (batch_size, sequence_length, d_model) (Tensor)

        Returns :
            y : RMS prop output of shape (batch_size, sequence_length, d_model) (Tensor)
        """
        ip_dtype = x.dtype
        # convert to float 32
        batch_f32 = x.to(torch.float32)
        batch_sq_f32 = batch_f32**2
        batch_mean = torch.mean(batch_sq_f32, dim=2)
        rms_dn = (batch_mean + self.eps) ** 0.5
        y = torch.einsum("... mno,... mn-> ... mno", x, rms_dn.reciprocal())
        if len(self.g_i.shape) < 2:
            self.g_i = nn.Parameter(self.g_i.unsqueeze(0))
        y = torch.einsum("mno,lo -> mno", y, self.g_i)
        # convert back to in dtype
        y = y.to(ip_dtype)
        return y
