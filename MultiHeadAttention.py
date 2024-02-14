import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image # 공부해보기
from torchvision.transforms import Compose, Resize, ToTensor # 공부해보기
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce # 공부해보기
from torchsummary import summary 
from einops import einsum
from patchembedding import PatchEmbedding

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_size: int = 768 # patch_size * patch_size  * 3 (채널 수 RGB) 
                 , head_num: int = 8
                 , dropout: float = 0
                 , mask: bool = None):
        
        super().__init__()
        # 주요 Variable
        self.embedding_size = embedding_size
        self.head_num = head_num
        self.scailing = (self.embedding_size//self.head_num)**(-0.5)
        
        # Learning Parameter 
        # self.values_layer = nn.Linear(torch.randn(self.embedding_size, self.embedding_size) # 실수했던 부분
        self.querys_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.keys_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.values_layer = nn.Linear(self.embedding_size, self.embedding_size)

        # 
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x: Tensor, mask: Tensor = None):

        # Input Tensor Shape: (b, n, embedding_size)
        # b: batch size, n: patch_num, h: head_size, d: embedding dimention / head_num

        querys = rearrange(self.querys_layer(x), "b n (h d) -> b h n d", h= self.head_num)
        keys = rearrange(self.keys_layer(x), "b n (h d) -> b h n d", h= self.head_num)
        values = rearrange(self.values_layer(x), "b n (h d) -> b h n d", h= self.head_num)

        # 아래의 형식에 맞춰서 써야한다.
        energy = torch.einsum( 'bhqd, bhkd -> bhqk', querys, keys)
        if mask is not None:
            energy.masked_fill(mask == 0, -1e09)
        # dim == -1 마지막 차원을 기준으로 한다. 
        # 즉, energy의 embedding 차원 768개에 대해서 softmax를 취한다. 
        weight_value = F.softmax(energy, dim= -1) * self.scailing # shape: (b, h, q, k) 
        weight_value = self.att_drop(weight_value)
        attention_value = torch.einsum('bhqk, bhkd -> bhqd', weight_value, values) # shape (b, h, n, d)
        attention_value = rearrange(attention_value, 'b h n d -> b n (h d)') # shape (b, n, embedding)

        # Ouput Tensor Shape: (b, n, embedding_size)
        return attention_value
    
x = torch.randn(100, 3, 224, 224)
embedding = PatchEmbedding()
patched = embedding(x)
attention = MultiHeadAttention()
value = attention(patched)
print(value.shape)    








