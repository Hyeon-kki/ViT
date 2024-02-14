import torch
from torch import nn
from Encoder import Encoder

class TransformerEncoder(nn.Sequential):

    def __init__(self, encoder_num: int = 12, **kwargs):

        super().__init__(
            # encoder_num 만큼 쌓는다. 
            *[Encoder(**kwargs) for _ in range(encoder_num)]
        )