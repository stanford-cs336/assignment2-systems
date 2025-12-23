import torch
import numpy as np


class flashAtten(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,  # (B,Nq,d)
        K: torch.Tensor,  # (B,Nk,d)
        V: torch.Tensor,  # (B,Nv,d)
        is_causal: bool,
    ):
        """
        Computes attention score for given Q,K and finds O.

        Args :
            Q - Query matrix (B,Nq,d)
            K - Key matrix (B,Nk,d)
            V - Value matrix (B,Nk,d)
            Bq - Tile size Q (A,A)
            Bk - Tile size K,V (B,B)
        Return
            O - Output projection
            L - Log Sum exponent
        """
        Bq = 16
        Bk = 16
        # compute the number of tiles of Q
        Tq = int(np.ceil(Q.shape[-2] / Bq))
        # compute the number of tiles of K,V
        Tk = int(np.ceil(K.shape[-2] / Bk))
        # slice the matrices
        sliced_Q = torch.reshape(Q, (Q.shape[0], Tq, Bq, Q.shape[-1]))
        sliced_K = torch.reshape(K, (Q.shape[0], Tk, Bk, K.shape[-1]))
        sliced_V = torch.reshape(V, (Q.shape[0], Tk, Bk, V.shape[-1]))
        # define the entire O
        O = torch.zeros_like(Q, dtype=torch.float32)
        # define the entire L
        L = torch.zeros((Q.shape[0], Q.shape[-2]))
        # compute O,P
        for i in range(Tq):
            print("i- ", i)
            current_Q = sliced_Q[:, i, :, :]
            # init m for online softmax
            m = -1 * torch.ones((current_Q.shape[0], current_Q.shape[1])) * torch.inf
            l = torch.zeros((current_Q.shape[0], current_Q.shape[1]))

            for j in range(Tk):
                current_K = sliced_K[:, j, :, :]
                current_V = sliced_V[:, j, :, :]
                # compute S
                S = current_Q @ current_K.transpose(-1, -2) / np.sqrt(Q.shape[-1])
                # compute m_i_j
                prev_m = m
                m = torch.maximum(prev_m, torch.max(S, dim=-1).values)
                # compute P
                P = torch.exp(S - m.unsqueeze(-1).expand(4, 16, 16))
                # compute l
                prev_l = l
                l = torch.exp(prev_m - m) * prev_l + torch.sum(P, dim=-1)
                # compute O
                prev_O = O[:, i * Bq : (i + 1) * Bq, :]
                O[:, i * Bq : (i + 1) * Bq, :] = (
                    torch.diag_embed(torch.exp(prev_m - m).squeeze(-1)) @ prev_O
                    + P @ current_V
                )
            # Denominator division of online softmax.This is one tile output
            O[:, i * Bq : (i + 1) * Bq, :] = O[
                :, i * Bq : (i + 1) * Bq, :
            ] / l.unsqueeze(-1)
            # log likelihood
            L[:, i * Bq : (i + 1) * Bq] = m + torch.log(l)

        ctx.save_for_backward(L)
        return O
