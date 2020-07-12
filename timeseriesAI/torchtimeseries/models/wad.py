import torch
from torch import nn
from torch.nn import functional as F

class WeightedAdataptivePooling1d(nn.Module):
    def __init__(self, output_size):
        super(WeightedAdataptivePooling1d, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(output_size)
        self.max = nn.AdaptiveMaxPool1d(output_size)
        self.p = nn.Parameter(torch.tensor([1/2, 1/2]))
        
    def forward(self, x):
        avg = self.avg(x)
        mx = self.max(x)
        w = F.softmax(self.p)
        return (w[0] * avg) + (w[1] * mx)
    
class WeightedPooling1d(nn.Module):
    def __init__(self, ks=2, stride=None, padding=0, ceil_mode=False):
        super(WeightedPooling1d, self).__init__()
        self.avg = nn.AvgPool1d(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)
        self.max = nn.MaxPool1d(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)
        self.p = nn.Parameter(torch.tensor([1/2, 1/2]))
        
    def forward(self, x):
        avg = self.avg(x)
        mx = self.max(x)
        w = F.softmax(self.p)
        return (w[0] * avg) + (w[1] * mx)
        