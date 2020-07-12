import torch
import torch.nn as nn
from .layers import *
from fastai.vision import spectral_norm
from .tae import *

__all__ = ['ResNetSA']

def _conv1d_spect(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in:int, ks=1, sym=False):
        super().__init__()
        self.sym,self.n_in = sym,n_in
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self,x):
        if self.sym:
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)

        size = x.size()
        x = x.view(*size[:2],-1)

        convx = self.conv(x)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


class ResBlock(nn.Module):
    def __init__(self, ni, nf, ks=[7, 5, 3], act_fn='relu'):
        super().__init__()
        self.conv1 = convlayer(ni, nf, ks[0], act_fn=act_fn)
        self.conv2 = convlayer(nf, nf, ks[1], act_fn=act_fn)
        self.conv3 = convlayer(nf, nf, ks[2], act_fn=False)
        self.sa = SimpleSelfAttention(nf, ks=1)

        # expand channels for the sum if necessary
        self.shortcut = noop if ni == nf else convlayer(ni, nf, ks=1, act_fn=False)
        self.act_fn = get_act_layer(act_fn)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sa(x)
        
        sc = self.shortcut(res)
        x += sc
        x = self.act_fn(x)
        return x
    
class ResNetSA(nn.Module):
    def __init__(self,c_in, c_out, pool=nn.AdaptiveAvgPool1d, temporal=False):
        super().__init__()
        nf = 64

        self.block1 = ResBlock(c_in, nf, ks=[7, 5, 3], act_fn='relu')
        self.block2 = ResBlock(nf, nf * 2, ks=[7, 5, 3], act_fn='relu')
        self.block3 = ResBlock(nf * 2, nf * 2, ks=[7, 5, 3], act_fn='relu')
        self.gap = pool(1)
        self.fc = nn.Linear(nf * 2, c_out)
        self.tea = TemporalAttentionEncoder(in_channels=nf*2, n_head=4, d_k=32, d_model=None, n_neurons=[4*nf*2, nf*2], dropout=0.2, T=1000, len_max_seq=24, positions=None) if temporal else noop

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.tea(self.gap(x)).squeeze(-1)
        return self.fc(x)