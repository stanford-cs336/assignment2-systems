import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from einops import rearrange, einsum


class Embedding(nn.Module):
    """
    Construct an embedding module. This function should accept the following parameters:
    Args:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters

    Returns:
        None
    """

    def __init__(
        self, num_embeddings, embedding_dim, weights=None, device=None, dtype=None
    ):
        super().__init__()
        self.vocab_size = num_embeddings
        self.emb_dim = embedding_dim

        # embedding is a learnable parameter
        if device.type == "cuda":  # pyright: ignore[reportOptionalMemberAccess]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device.type == "mps":  # pyright: ignore[reportOptionalMemberAccess]
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        else:
            self.device = "cpu"

        if weights != None:
            self.embeddings = weights
            self.embeddings.to(device)
        else:
            self.embeddings = nn.Parameter(
                torch.empty(self.vocab_size, self.emb_dim, device=device, dtype=dtype)
            )
            init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor, device=None) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            self: Class object ( Embeddings )
            token_ids : List of token ids that needs to be converted ( 2d Tensor )
        Returns:
            embed_mat = input tokens converted to embedding vector ( 3d tensor)
        """
        embed_mat = torch.empty(
            token_ids.shape[0], token_ids.shape[1], self.emb_dim, device=device
        )
        for b in range(token_ids.shape[0]):
            batch = token_ids[b]
            embed_mat[b] = self.embeddings[batch]
        return embed_mat
