import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torch import Tensor
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        # shallownet（浅层神经网络）。本模型用了一个结构较为简单的卷积神shallownet替代Transformer中的位置嵌入。
        self.shallownet = nn.Sequential(
            # Conv2d(1=输入特征矩阵深度，40=卷积核个数，(1,25)=卷积核尺寸，(1,1)=卷积核横纵向移动步长，不指定padding=不进行填充)
            # 这里将全部的40通道改为50通道，以期望增强观测的结果。
            nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # 注释以下代码并且拆分Conv与AvgPool，观察各个输出的尺寸。
            nn.Conv2d(8, 64, kernel_size=(62, 2), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(64),   # 批量归一化的输入通道数为40，对应上面Conv2d的40个输出通道。
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 4)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
                                             # 池化核大小为(1,75)，移动步长为(1,15)
            nn.Dropout(0.5),
        )

        # self.shallownet2 = nn.Sequential(
        #     nn.Conv2d(40, 40, (22, 1), (1, 1)),
        # )

        # self.shallownet3 = nn.Sequential(
        #     nn.BatchNorm2d(40),
        #     nn.ELU(),
        #     nn.AvgPool2d((1,75),(1,15)),
        #     nn.Dropout(0.5),
        # )

        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # print('input shape of patch embedding: ',x.shape)
        # input shape of patch embedding:  torch.Size([144, 1, 22, 751])

        x = self.shallownet(x)  # 128, 1, 124, 32 -> 128, 64, 1, 7

        x = self.projection(x)
        # print('after projection, the shape of x: ',x.shape)
        # after projection, the shape of x:  torch.Size([144, 44, 40])

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(448, 256),    # 原始是2440。这里进行改动，换成了forward中x的尺寸(72,1760)
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=64, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

