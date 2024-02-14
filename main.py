import torch
from torch import nn
from torchsummary import summary
from patchembedding import PatchEmbedding
from TransformerEncoder import TransformerEncoder
from head import ClassfierHead

class ViT(nn.Sequential):

    def __init__(self,  in_channel: int = 3,
                        patch_size: int = 16,
                        embedding_size: int = 768,
                        img_size: int = 224,
                        depth:int = 6):


        super().__init__(
            PatchEmbedding(in_channel= in_channel, patch_size= patch_size, embedding= embedding_size, img_size= img_size),
            TransformerEncoder(encoder_num= depth, embedding_size =embedding_size),
            ClassfierHead(embedded_size= embedding_size, class_num= 1000)
        )

vit_model = ViT()
summary(model= vit_model, input_size = (3, 224, 224), device='cpu')