from collections.abc import Callable, Iterable
from typing import Optional
import torch
import torch.optim as optim
import math


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = {"lr": lr, "beta": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta, eps, lamda = (
                group["lr"],
                group["beta"],
                group["eps"],
                group["weight_decay"],
            )  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Get state associated with p.
                state = self.state[p]
                # Get iteration number from the state, or initial value.
                if state.get("t", 0) == 0:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)  # type: ignore
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)  # type: ignore
                # get current m/v
                m = state["m"]
                v = state["v"]

                state["t"] = state.get("t", 0) + 1  # Increment iteration number.
                t = state["t"]

                g = p.grad.data  # Get the gradient of loss with respect to p.
                # update moments
                state["m"] = beta[0] * m + (1 - beta[0]) * g
                state["v"] = beta[1] * v + (1 - beta[1]) * (g**2)
                # alpha_t computation
                lr_t = lr * math.sqrt(1 - (beta[1]) ** t) / (1 - (beta[0] ** t))
                # Update weight tensor in-place.
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * lamda * p.data
