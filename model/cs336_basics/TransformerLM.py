from cs336_basics import Transformer
from cs336_basics import RMSNorm
from cs336_basics import Linear
from cs336_basics import Embedding

import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """
    TransformerLM implements a language model, takes the input and converts to embeddings based on context
    length.
    @inparam[0] : vocab_size:The size of the vocabulary, necessary for determining the dimensionality of the token
                            embedding matrix (int)
    @inparam[1] : context_length:The maximum context length, necessary for determining the dimensionality of
                                the position embedding matrix (int)
    @inparam[2] : num_layers:The number of Transformer blocks to use (int)
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        device=None,
    ):
        super().__init__()
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        # Define transformer
        self.Tr = nn.ModuleList(
            [
                Transformer.Transformer(d_model, num_heads, d_ff, device=self.device)
                for _ in range(num_layers)
            ]
        )
        # Define post transformer norm
        self.post_norm = RMSNorm.RMSNorm(d_model, device=self.device)
        # Define Linear layer
        self.L = Linear.Linear(d_model, vocab_size, device=self.device)
        # Define Embedding layer
        self.E = Embedding.Embedding(vocab_size, d_model, device=self.device)

    def tranform_lm_model(self, x, rope_theta, token_positions, max_seq_len):
        # Run embedding module
        t_in = self.E.forward(x, self.device)
        for l in range(self.num_layers):
            y = self.Tr[l].transform(
                t_in,
                theta=rope_theta,
                token_positions=token_positions,
                max_seq_len=max_seq_len,
                device=self.device,
            )  # pyright: ignore[reportCallIssue]
            t_in = y
        rms_trf = self.post_norm(y)
        fc_y = self.L.forward(rms_trf)

        return fc_y
