import torch


def softmax(
    x: torch.tensor, dim: int  # type: ignore
) -> torch.tensor:  # pyright: ignore[reportGeneralTypeIssues]
    x_d_norm = x - torch.max(x, dim=dim, keepdim=True).values
    y = torch.exp(x_d_norm) / torch.sum(torch.exp(x_d_norm), dim=dim, keepdim=True)
    return y
