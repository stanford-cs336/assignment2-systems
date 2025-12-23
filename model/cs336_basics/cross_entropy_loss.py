from .softmax import softmax
import torch


def cross_entropy_loss(logits: torch.Tensor, x: torch.Tensor):
    """
    Computes the cross entropy loss L(theta,D)
        L(Theta,D) = sum_{x E D} sum_{i :1 -> m} -log(softmax(o_i[x_i+1]))
                    -------------------------------------------------------
                                    |D|m
        where   o_i[x_i+1] : logits for x_i+1
                |D| : Data consisting of sequences of length 'm'.
    Args
        logits  :   logits for each x_i in D,(Float[Tensor, "batch_size vocab_size"])
        x       :   targets x_t+1,(Int[Tensor, "batch_size"])
    """
    # extract Values (a)
    logits_at_target = -torch.gather(
        logits, dim=len(logits.shape) - 1, index=x.unsqueeze(-1)
    ).squeeze(-1)
    # max across every BxT (b)
    logits_max_batch = torch.max(logits, dim=2).values
    # compute norm term
    norm_term = torch.sum(
        torch.exp(
            logits
            - logits_max_batch.unsqueeze(2).expand(
                logits.shape[0], logits.shape[1], logits.shape[2]
            )
        ),
        dim=2,
    )
    loss = torch.sum(logits_at_target + logits_max_batch + torch.log(norm_term)) / (
        logits.shape[0] * logits.shape[1]
    )

    return loss
