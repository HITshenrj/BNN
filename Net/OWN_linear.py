import torch
import torch.nn as nn
from .OWN import OWNNorm


class OWNLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(OWNLinear, self).__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(out_dim, ))
        self.own_ = OWNNorm(norm_groups=1)

    def forward(self, x, own=True):
        if own:
            self.W.data = self.own_(self.W).view(self.W.data.size(0), -1)

        out = torch.matmul(x, self.W) + self.bias
        return out

    def back_ward(self, x):

        out = torch.matmul(x - self.bias, self.W.T)

        return out
