import torch
import torch.nn as nn


# Creating a custom GLU activation function class
class GLUActivation(nn.Module):
    def __init__(self, slope: float = 0.3):
        super().__init__()
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the input into two equal parts (chunks) along dimension 1
        out, gate = x.chunk(2, dim=1)
        
        # Perform element-wise multiplication of the first half (out)
        # with the result of applying LeakyReLU on the second half (gate)
        x = out * self.lrelu(gate)
        return x
    
