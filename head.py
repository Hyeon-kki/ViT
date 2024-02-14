import torch
from torch import nn
from einops.layers.torch import Reduce
class ClassfierHead(nn.Sequential):

    def __init__(self, embedded_size:int = 768, class_num:int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction= "mean"),
            nn.LayerNorm(embedded_size),
            nn.Linear(embedded_size, class_num),
        )
