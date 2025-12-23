import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math
from einops import einsum
from .Linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model, device=None):
        super().__init__()
        self.d_model = d_model
        self.dff = self.d_model * 8 / 3
        n_int = math.ceil(self.dff / 64)
        self.dff = n_int * 64
        self.fc1 = Linear(self.d_model, self.dff, device=device)
        self.fc2 = Linear(self.dff, self.d_model, device=device)
        # Swish + GLU
        self.glu_w = nn.Parameter(torch.empty(self.dff, self.d_model, device=device))
        init.trunc_normal_(self.glu_w, mean=0, std=1, a=-3 * 1, b=3 * 1)

    def s_glu(self, x, a1):
        # dff x 1
        w3_x = einsum(x, self.glu_w, "batch time feat,out feat->batch time out")
        # silu(x) dffx1
        silu_x = a1 * F.sigmoid(a1)
        # glu dff x 1
        h1 = silu_x * w3_x
        return h1

    def forward(self, x):
        a1 = self.fc1.forward(x)
        h1 = self.s_glu(x, a1)
        y = self.fc2.forward(h1)
        return y
