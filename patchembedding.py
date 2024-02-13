import torch
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

x = torch.randn(1000, 3, 224, 224)

## Image를 이미지의 크기/패치의 크기의 패치로 쪼개고 Flatten 시킨다. 
p = 16 # 논문에서 16이 계산 비용대비 효율적이라 소개한다. 

## 이때, p_h(p_w): 패치의 크기 and n_h(n_w): 패치의 크기 나눴을 때 가로 수 
patches = rearrange(x, 'b c (p_h n_h) (p_w n_w) -> b (n_h n_w) (p_h p_w c)', p_h=p, p_w=p)

class PatchEmbedding(nn.Module):

    # defalut는 Paper를 따라간다.
    def __init__(self, in_channel: int = 3, patch_size: int = 16, embedding: int = 768, img_size: int = 224):
        super(PatchEmbedding, self).__init__()
        self.patch_num = (img_size//patch_size)**2
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embedding = embedding

        # nn.Parameter는 PyTorch에서 모델의 '학습 가능한 파라미터'를 정의하는 데 사용되는 클래스입니다. 
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding))

        # Position embedding에서 +1이 되는 이유는 cls token 때문이다.
        self.postions = nn.Parameter(torch.randn(self.patch_num+1, self.embedding))

        self.projection = nn.Sequential(
            # con2d말고도 Linear로 출력값을 맞출 수 있다. 하지만, con2d가 성능이 더 좋다고 말한다.
            nn.Conv2d(in_channel, embedding, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e') # 변환하는게 약하다. 보완하기!!!!!!!!!!!!!
        )
        # Linear 층이다.
        # self.projection = nn.Sequential(
        #     # break-down the image in s1 x s2 patches and flat them
        #     Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
        #     nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        #### Add cls token 
        cls_tokens = repeat(self.cls_token,'() n e -> b n e', b = x.shape[0])
        # print(x.shape) # torch.Size([1000, 196, 768])
        x = torch.cat([cls_tokens, x], dim=1)
        # print(x.shape) # torch.Size([1000, 197, 768])
        
        #### Add Position Embedding  
        print(self.postions.shape)
        # 텐서끼리 더할때 행과 열이 일치한다면 예상과 동일하게 더하기를 진행한다. 
        x += self.postions 
        return x




x = PatchEmbedding()(x)
# print(x.shape)

