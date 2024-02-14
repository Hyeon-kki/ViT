import torch
from torch import nn 

class FeedForwardBlock(nn.Module):

    def __init__(self, embedding_size: int, expansion: int = 4, drop_ratio: int = 0):
        super().__init__()
        self.embedding_size = embedding_size
        self.expansion = expansion
        self.drop_ratio = drop_ratio

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * self.expansion),
            nn.GELU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.embedding_size * self.expansion, self.embedding_size))

    def forward(self, x):
        output = self.projection(x)
        return output

