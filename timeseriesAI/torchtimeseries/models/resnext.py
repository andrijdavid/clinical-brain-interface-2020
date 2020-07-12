# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Wang, Z., Yan, W., & Oates, T. (2017, May). Time series classification from scratch with deep neural networks: A strong baseline. In 2017 international joint conference on neural networks (IJCNN) (pp. 1578-1585). IEEE.

# Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4), 917-963.
# Official ResNet TensorFlow implementation: https://github.com/hfawaz/dl-4-tsc

# 👀 kernel filter size 8 has been replaced by 7 (I believe it's a bug)


import torch
import torch.nn as nn
from .layers import *

__all__ = ['ResNeXt']

class ResBlock(nn.Module):
    def __init__(self, ni, nf, cardinality=12, stride=1, ks=[1,3,1], act_fn='relu'):
        super().__init__()
        self.conv1 = convlayer(ni, nf, ks[0], act_fn=act_fn, bias=False)
        self.conv2 = convlayer(nf, nf, ks[1], act_fn=act_fn, stride=stride, bias=False, padding=1, groups=cardinality)
        self.conv3 = convlayer(nf, nf, ks[2], act_fn=False, bias=False)

        # expand channels for the sum if necessary
        self.shortcut = noop if ni == nf else convlayer(ni, nf, ks=1, act_fn=False, stride=stride, bias=False)
        self.act_fn = get_act_layer(act_fn)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        sc = self.shortcut(res)
        x += sc
        x = self.act_fn(x)
        return x
    
class ResNeXt(nn.Module):
    def __init__(self,c_in, c_out, cardinality=2, bottleneck_width=64, pool=nn.AdaptiveAvgPool1d):
        super().__init__()
        nf = cardinality * bottleneck_width
        self.block1 = ResBlock(c_in, nf, act_fn='relu', stride=1, cardinality=cardinality)
        self.block2 = ResBlock(nf, nf * 2, act_fn='relu', stride=2, cardinality=cardinality)
        self.block3 = ResBlock(nf*2, nf*2*2, act_fn='relu', stride=2, cardinality=cardinality)
        self.gap = pool(1)
        self.fc = nn.Linear(nf*2*2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)