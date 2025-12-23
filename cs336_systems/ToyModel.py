import torch
import torch.nn as nn
import random


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        fc1 = self.fc1(x)
        print(f"data type of fc1 - {fc1.dtype}")
        x = self.relu(fc1)
        x = self.ln(x)
        print(f"data type of layer norm - {x.dtype}")
        x = self.fc2(x)
        print(f"data type of fc2 - {x.dtype}")
        return x


if __name__ == "__main__":
    M = ToyModel(8, 4)
    device = torch.device("cpu")

    x = torch.rand(4, 8, dtype=torch.float32, device=device)
    target = torch.randint(low=0, high=4, size=(4,))
    with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=False):
        # show the parameter and its dtypes
        print("#### Model params ####")
        for par in M.parameters():
            print(f"{par.dtype}")
        y = M.forward(x)
        print(f"Output y data type - {y.dtype}")
        """
        • the model parameters within the autocast context,
        • the output of the first feed-forward layer (ToyModel.fc1),
        • the output of layer norm (ToyModel.ln),
        • the model’s predicted logits,
        • the loss,
        • and the model’s gradients?
        """
    # define softmax
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y, target)
    print(f"Model loss - {loss.dtype}")
    loss.backward()
    print("### model param gradient types ###")
    for p in M.parameters():
        print(f"{p.grad.dtype}")  # type: ignore
