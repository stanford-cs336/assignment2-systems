import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.mha import MHA
from cs336_basics.SwiGLU import SwiGLU


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # Define Pre Norm Object for MHA
        self.RMS_MHA = RMSNorm(self.d_model, device=device)
        # Define pre norm for FFNN
        self.RMS_FF = RMSNorm(self.d_model, device=device)
        # Define MHA object
        self.mha = MHA(self.d_model, self.num_heads, device=device)
        # Define FF NN
        self.FF = SwiGLU(self.d_model, device=device)

    def transform(
        self, x, token_positions=None, theta=None, max_seq_len=None, device=None
    ):
        rms_mha = self.RMS_MHA.forward(x)
        x_post_mha = x + self.mha.mha_self_attention(
            rms_mha, theta, token_positions, max_seq_len, device=device
        )

        rms_ff = self.RMS_FF.forward(x_post_mha)
        x_post_ffnn = x_post_mha + self.FF.forward(rms_ff)
        return x_post_ffnn
