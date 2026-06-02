import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256, bias=False)
        self.ln = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, out_features, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"fc1(x) dtype: {x.dtype}")
        x = self.ln(x)
        print(f"ln(x) dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"fc2(x) dtype: {x.dtype}")
        return x

def main():
    device = "mps"
    torch.manual_seed(42)
    x = torch.randn((1024, 1024), dtype=torch.float32, device=device)
    print(f"x: {x[0, 0:10]}")
    print(f"x dtype: {x.dtype}")
    dtype : torch.dtype = torch.float16
    model : torch.nn.Module = ToyModel(1024, 1024).to(device=device)
    y1 = model(x)
    print(f"model(x) dtype: {y1.dtype}")
    print(f"y1: {y1[0, 0:10]}")
    print("down scale")
    with torch.autocast(device_type="mps", dtype=dtype):
        y = model(x)
        print(f"model(x) dtype: {y.dtype}")
        print(f"y: {y[0, 0:10]}")
        
if __name__ == "__main__":
    main()
    
