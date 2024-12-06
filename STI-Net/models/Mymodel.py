import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from einops import rearrange
from torch import einsum
from typing import Tuple

sys.path.append('./models')

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

class myconv(nn.Module):
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

        self.block1_1 = nn.Conv3d(k1_inout[0], k1_inout[1], kernel_size=(3,3,33), stride=(2,2,1), padding=(2,2,16), groups=k1_inout[0], bias=bias)
        self.block1_2 = nn.Conv3d(k1_inout[0], k1_inout[1], kernel_size=(3,3,65), stride=(2,2,1), padding=(2,2,32), groups=k1_inout[0], bias=bias)

        self.bn_elu_drop_1 = nn.Sequential(nn.BatchNorm3d(num_features=k1_inout[1]*2), 
                                           nn.ELU(), 
                                           nn.Dropout(dropout))

        self.block2_1 = Conv2dWithL2Norm(k1_inout[1]*2, k1_inout[1]*8, kernel_size=(6,1), stride=(4,1), padding=(1,0), groups=k1_inout[1]*2)
        self.bn_elu_drop_avgpool = nn.Sequential(nn.BatchNorm2d(num_features=k1_inout[1]*8), 
                                                 nn.ELU(), 
                                                 nn.AvgPool2d((1,2), stride=(1,2)),
                                                 nn.Dropout(dropout))
        
        self.block2_3 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,3), stride=1, padding=(1,1), bias=bias)
        self.block2_4 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(5,3), stride=1, padding=(2,1), bias=bias)
        self.block2_5 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(9,3), stride=1, padding=(4,1), bias=bias)

        self.block2_6 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,3), stride=1, padding=(1,1), bias=bias)
        self.block2_7 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,5), stride=1, padding=(1,2), bias=bias)
        self.block2_8 = nn.Conv2d(k1_inout[1]*8, k1_inout[1]*8, kernel_size=(3,9), stride=1, padding=(1,4), bias=bias)

        self.block3_1 = Conv2dWithL2Norm(k1_inout[1]*8, k1_inout[1]*4, kernel_size=(16,1), stride=1)

        self.block3_2 = nn.Conv2d(k1_inout[1]*4, k1_inout[1]*4, kernel_size=(1,33), stride=(1,1), padding=(0,16), bias=bias)
        self.bn_elu_drop_2 = nn.Sequential(nn.BatchNorm2d(num_features=k1_inout[1]*4),
                                           nn.ELU(),
                                           nn.Dropout(dropout))

    
    def forward(self, x):  # input: 64, 1, 32, 32, 32: sample_num, conv_channel, width, height, time
        x1_1 = self.block1_1(x) 
        x1_2 = self.block1_2(x)

        x1_3 = torch.cat((x1_1, x1_2), dim=1)            # 64, 16, 8, 8, 32
        x1_3 = self.bn_elu_drop_1(x1_3)

        x1 = rearrange(x1_3, 's c w h t -> s c (w h) t') # 64, 16, 64, 32

        x2_1 = self.block2_1(x1)                         # 64, 64, 16, 32
        x2_1 = self.bn_elu_drop_avgpool(x2_1)            # 64, 64, 16, 16

        # inception block for spatial dimension
        x2_2 = self.block2_3(x2_1)
        x2_3 = self.block2_4(x2_1)
        x2_4 = self.block2_5(x2_1)                       # 64, 64, 16, 16
        # x2 = torch.cat((x2_2, x2_3), dim=1)
        x2_5 = x2_2 + x2_3 + x2_4                        # 64, 64, 16, 16  last dim is still time dimension.
        x2_5 = self.elu_drop(x2_5)

        # inception block for time dimension
        x2_6 = self.block2_6(x2_5)
        x2_7 = self.block2_7(x2_5)
        x2_8 = self.block2_8(x2_5)
        x2 = x2_6 + x2_7 + x2_8
        x2 = self.elu_drop(x2)                           # 64, 64, 16, 16
    
        x3_1 = self.block3_1(x2)                         # 64, 32, 1, 16
        x3_2 = self.block3_2(x3_1)
        x3_2 = self.bn_elu_drop_2(x3_2)
        x3 = x3_2                                        # 64, 32, 1, 16

        return x3

class STINet(nn.Module):
    def __init__(self,
                 emb_dim,
                 category_num = 6):
        super().__init__()
        
        self.linear1 = nn.Linear(emb_dim*16, 128)
        self.linear2 = nn.Linear(128, category_num)
        
        self.myconv = myconv()

        # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x_batch):

        # if use modified network:
        x_batch = rearrange(x_batch, 's (o t) w h  -> s o w h t', o=1)
        flash_input = self.myconv(x_batch)
        flash_output = flash_input.flatten(start_dim=1)  # 64, 16*32

        final_output = self.linear2(self.linear1(flash_output))
        
        return final_output
        