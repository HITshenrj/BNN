import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable


#  norm funcitons--------------------------------


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input


class OWNNorm(torch.nn.Module):
    def __init__(self, norm_groups=1, *args, **kwargs):
        super(OWNNorm, self).__init__()
        self.norm_groups = norm_groups

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):   # Forward pass of Orthogonal Linear Module
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  #   type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)     # Zc= Z − (1/d)Z1_d(1_d.T)
        S = torch.matmul(Zc, Zc.transpose(1, 2))      # S = Zc * Zc.T
        wm = torch.randn(S.shape).to(S) 
        for i in range(self.norm_groups):
            U, Eig, _ = S[i].svd()
            Scales = Eig.rsqrt().diag()
            wm[i] = U.mm(Scales).mm(U.t())     # P = DΛ(−1/2)D.T
        W = wm.matmul(Zc)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['OWN:']
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


if __name__ == '__main__':
    oni_ = OWNNorm(norm_groups=1)
    print(oni_)
    w_ = torch.randn(2, 12)
    print(w_)
    w_.requires_grad_()
    y_ = oni_(w_)
    print(y_)
    z_ = y_.view(w_.size(0), -1)
    print(z_)
    print(z_.matmul(z_.t()))

    print(y_.sum().backward())
    print('w grad', w_.grad.size())

    a = torch.tensor([1., 1.])
    print('a', a)
    b = torch.matmul(a, z_)
    print('b', b)
    print('inverse', torch.matmul(b, z_.T))
