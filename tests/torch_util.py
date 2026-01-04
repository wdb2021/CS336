import torch
import torch.nn as nn
from a_myLM import LinearModel


def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(index))
        return torch.device(f"cuda:{index}")
    else:
        print("No GPU available, using CPU instead.")
        return torch.device("cpu")

get_device()

class MLP(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([LinearModel(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)

