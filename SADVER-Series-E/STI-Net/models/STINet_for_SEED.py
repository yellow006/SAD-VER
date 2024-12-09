import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Tuple

"""
STI-Net (Spatial-Temporal Inception Net): A compact convolutional neural network with 2 different Inception structure.
STI Net accepts a 3-D input: (batchsize, num_samples(t), width(w), height(h)).
for example (ignore MLPs), input: (64, 32, 13, 13) ---> output: (64, 512).   512 stands for the feature dim.
"""

class Conv2dWithL2Norm(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithL2Norm, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithL2Norm, self).forward(x)


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True):
        super().__init__()
        self.depth = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size=(kernel_size,1),
                               stride=stride,
                               padding=padding,
                               groups=in_channels,
                               bias=bias)
        self.point = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth(x)
        x = self.point(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, 
                 size_per_sample: Tuple[int, int]=[32, 32],
                 input_samples = 32,
                 k1_inout: Tuple[int, int]=[1, 8],
                 dropout = 0.25,
                 bias: bool = False):
        super().__init__()
        self.size_per_sample = size_per_sample
        self.samples = input_samples
        self.elu_drop = nn.Sequential(nn.ELU(), nn.Dropout(dropout))

        self.block1_1 = nn.Conv3d(k1_inout[0], k1_inout[1], kernel_size=(6,6,2), stride=(1,1,1), padding=(0,0,0), groups=k1_inout[0], bias=bias)
        self.block1_2 = nn.Conv3d(k1_inout[0], k1_inout[1], kernel_size=(5,5,4), stride=(2,2,1), padding=(1,1,1), groups=k1_inout[0], bias=bias)

        self.bn_elu_drop_1 = nn.Sequential(nn.BatchNorm3d(num_features=k1_inout[1]*2), 
                                           nn.ELU(), 
                                           nn.Dropout(dropout))

        self.block2_1 = Conv2dWithL2Norm(k1_inout[1]*2, k1_inout[1]*8, kernel_size=(4,1), stride=(2,1), padding=(1,0), groups=k1_inout[1]*2)
        self.bn_elu_drop_avgpool = nn.Sequential(nn.BatchNorm2d(num_features=k1_inout[1]*8), 
                                                 nn.ELU(), 
                                                 nn.AvgPool2d((1,2), stride=(1,2)),
                                                 nn.Dropout(dropout))
        
        self.block2_3 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(1,3), stride=1, padding=(0,1), bias=bias)
        self.block2_4 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,3), stride=1, padding=(1,1), bias=bias)
        self.block2_5 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(5,3), stride=1, padding=(2,1), bias=bias)

        self.block2_6 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,1), stride=1, padding=(1,0), bias=bias)
        self.block2_7 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,3), stride=1, padding=(1,1), bias=bias)
        self.block2_8 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,5), stride=1, padding=(1,2), bias=bias)

        self.block3_1 = Conv2dWithL2Norm(k1_inout[1]*8, k1_inout[1]*4, kernel_size=(8,1), stride=1)
        self.block3_2 = nn.Conv2d(k1_inout[1]*4, k1_inout[1]*4, kernel_size=(1,3), stride=(1,1), padding=(0,1), bias=bias)

        self.bn_elu_drop_2 = nn.Sequential(nn.BatchNorm2d(num_features=k1_inout[1]*4),
                                           nn.ELU(),
                                           nn.Dropout(dropout))

    
    def forward(self, x):
        x1_1 = self.block1_1(x)    # 64 1 9 9 5 -> 64 8 4 4 4
        x1_2 = self.block1_2(x)

        x1_3 = torch.cat((x1_1, x1_2), dim=1)
        x1_3 = self.bn_elu_drop_1(x1_3)

        x1 = rearrange(x1_3, 's c w h t -> s c (w h) t')  # 64 16 4 4 4 -> x1: 64 16 16 4

        x2_1 = self.block2_1(x1)   # 64 64 8 4
        x2_1 = self.bn_elu_drop_avgpool(x2_1)

        # inception block for spatial dimension
        x2_2 = self.block2_3(x2_1)
        x2_3 = self.block2_4(x2_1)
        x2_4 = self.block2_5(x2_1)
        x2_5 = x2_2 + x2_3 + x2_4
        x2_5 = self.elu_drop(x2_5)

        # inception block for time dimension
        x2_6 = self.block2_6(x2_5)
        x2_7 = self.block2_7(x2_5)
        x2_8 = self.block2_8(x2_5)
        x2 = x2_6 + x2_7 + x2_8
        x2 = self.elu_drop(x2)   # x2: 64 64 8 2
    
        x3_1 = self.block3_1(x2)
        x3_2 = self.block3_2(x3_1)
        x3 = self.bn_elu_drop_2(x3_2)  # x3：64 32 1 2

        return x3

class STINet(nn.Module):
    def __init__(self,
                 category_num = 3,
                 dropout = 0.3):
        super().__init__()
        
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, category_num)
        
        self.convencoder = ConvEncoder(dropout=dropout)
        
    def forward(self, x_batch):

        x_batch = rearrange(x_batch, 's (o t) w h  -> s o w h t', o=1)
        conv_output = self.convencoder(x_batch)
        conv_output = conv_output.flatten(start_dim=1)

        final_output = self.linear2(self.linear1(conv_output))
        
        return final_output

if __name__ == '__main__':
    input = torch.randn(64,5,9,9)
    model1 = STINet()
    output = model1(input)
    print(output.shape)