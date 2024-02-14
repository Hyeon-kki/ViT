import torch
from torch import nn
from torch import Tensor
from patchembedding import PatchEmbedding
from ResidualConnection import ResidualConnection
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForwardBlock

class Encoder(nn.Sequential):
    
    def __init__(self, embedding_size, 
                 expansion: int = 4,
                 drop_ratio: float = 0,
                 **kwargs):

        # Sequential을 상속받았기 때문에 바로 생성자 입력 
        super().__init__(
            ResidualConnection(nn.Sequential(
                # 정규화를 앞단에서 해줌
                nn.LayerNorm(embedding_size),
                MultiHeadAttention(embedding_size, **kwargs),
                nn.Dropout())
            ),
            ResidualConnection( nn.Sequential(
                nn.LayerNorm(embedding_size),
                FeedForwardBlock(embedding_size=embedding_size, expansion=expansion, drop_ratio= drop_ratio),
                nn.Dropout(drop_ratio))
            ),
        )

x = torch.randn(100, 3, 224, 224)
###### 이거 실수많다.
embedding = PatchEmbedding() 
patched = embedding(x)
encoder = Encoder(embedding_size=768, expansion=4, drop_ratio=0)
output = encoder(patched)
print(output.shape)