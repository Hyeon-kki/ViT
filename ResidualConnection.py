import  torch
from torch import nn
from torch import Tensor

class ResidualConnection(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.fn = layers

    def forward(self, x: Tensor) -> Tensor:
        value = self.fn(x)
        res = x
        output = value + x
        return output 