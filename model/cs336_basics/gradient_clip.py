import numpy
import torch


@torch.no_grad()
def gradient_clipping(params, max_l2_norm):
    """
    Performs gradient clipping w.r.t to max(l2 norm)
    args :
        params : list of params
    return :
        params : normalized list of params
    """
    eps = 1e-6
    grad = []
    grad = [
        p.grad.detach().float().norm(2).item() for p in params if p.grad is not None
    ]
    l2_g = numpy.linalg.norm(grad, 2)
    if l2_g > max_l2_norm:
        for m in range(len(params)):
            if not (params[m].grad is None):
                params[m].grad.mul_(max_l2_norm / (l2_g + eps))
