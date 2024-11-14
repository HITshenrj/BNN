"""
RealNVP
"""

import torch
import torch.nn as nn



class LayerNormFlow(nn.Module):

    def __init__(self, M, N, eps=1e-5):
        super(LayerNormFlow, self).__init__()
        self.log_gamma = nn.Parameter(torch.zeros(M+N))
        self.beta = nn.Parameter(torch.zeros(M+N))
        self.eps = eps

    def forward(self, M, N, invert=False):

        inputs = torch.cat([M, N], dim=-1)
        if not invert:

            self.batch_mean = inputs.mean(-1)

            self.batch_var = (
                                     inputs - self.batch_mean.reshape(-1, 1)).pow(2).mean(-1) + self.eps

            mean = self.batch_mean
            var = self.batch_var
            # print(mean)
            # print(var)

            x_hat = (inputs - mean.reshape(-1, 1)) / var.sqrt().reshape(-1, 1)
            # print(x_hat)
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            # print(y)
            # print('--------------')
            return y[:, :M.shape[-1]], y[:, -N.shape[-1]:]
        else:
            mean = self.batch_mean
            var = self.batch_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt().reshape(-1, 1) + mean.reshape(-1, 1)

            return y[:, :M.shape[-1]], y[:, -N.shape[-1]:]


class CoupleLayer(nn.Module):
    def __init__(self, M, N, fn, bn, key=0, reverse=False):
        super(CoupleLayer, self).__init__()
        self.reverse = reverse

        self.S = nn.Sequential()
        self.T = nn.Sequential()
        self.key = key

        if not reverse:
            for i in range(len(fn) + 1):
                if i == 0:
                    self.S.add_module('layer_f{}'.format(i), nn.Linear(M, fn[0]))
                    self.T.add_module('layer_f{}'.format(i), nn.Linear(M, fn[0]))
                    self.S.add_module('Activation_f{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_f{}'.format(i), nn.LeakyReLU(0.5))
                elif i < len(fn):
                    self.S.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], fn[i]))
                    self.T.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], fn[i]))
                    self.S.add_module('Activation_f{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_f{}'.format(i), nn.LeakyReLU(0.5))
                elif i == len(fn):
                    self.S.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], N))
                    self.T.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], N))
                    # self.S.add_module('Activation_f{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_f{}'.format(i), nn.Tanh())

        else:
            for i in range(len(bn) + 1):
                if i == 0:
                    self.S.add_module('layer_b{}'.format(i), nn.Linear(N, bn[0]))
                    self.T.add_module('layer_b{}'.format(i), nn.Linear(N, bn[0]))
                    self.S.add_module('Activation_b{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_b{}'.format(i), nn.LeakyReLU(0.5))
                elif i < len(bn):
                    self.S.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], bn[i]))
                    self.T.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], bn[i]))
                    self.S.add_module('Activation_b{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_b{}'.format(i), nn.LeakyReLU(0.5))
                elif i == len(bn):
                    self.S.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], M))
                    self.T.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], M))
                    # self.S.add_module('Activation_b{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_b{}'.format(i), nn.Tanh())
        self._init_weight()
        # print(self.S)
        # print(self.T)
    def _init_weight(self):
        for i in range(len(self.S)):
            if i % 2 == 0:
                nn.init.xavier_uniform_(self.S[i].weight)
                nn.init.zeros_(self.S[i].bias)
        for i in range(len(self.T)):
            if i % 2 == 0:
                nn.init.xavier_uniform_(self.T[i].weight)
                nn.init.zeros_(self.T[i].bias)

    def forward(self, input_M, input_N, invert=False):
        if not invert:
            if not self.reverse:
                output_M = input_M
                if self.key == 0:
                    S_out = torch.exp(self.S(input_M))
                elif self.key == 1:
                    S_out = -torch.exp(self.S(input_M))
                T_out = self.T(input_M)
                # print(input_N.shape)
                # print(S_out.shape)
                # print(T_out.shape)
                # print("__________________")
                output_N = input_N * S_out + T_out
            else:
                output_N = input_N
                if self.key == 0:
                    S_out = torch.exp(self.S(input_N))
                elif self.key == 1:
                    S_out = -torch.exp(self.S(input_N))

                T_out = self.T(input_N)
                output_M = input_M * S_out + T_out
        else:
            if self.reverse:
                output_N = input_N
                if self.key == 0:
                    S_out = torch.exp(-self.S(input_N))
                elif self.key == 1:
                    S_out = -torch.exp(-self.S(input_N))
                T_out = self.T(input_N)
                output_M = (input_M - T_out) * S_out
            else:
                output_M = input_M
                if self.key == 0:
                    S_out = torch.exp(-self.S(input_M))
                elif self.key == 1:
                    S_out = -torch.exp(-self.S(input_M))
                T_out = self.T(input_M)
                output_N = (input_N - T_out) * S_out

        return output_M, output_N


class RealNVP(nn.Module):
    def __init__(self, M, N, layers, fn, bn):
        super(RealNVP, self).__init__()

        self.M = M
        self.N = N
        self.Layers = layers
        self.Couple_Layers = nn.ModuleList()
        for i in range(layers):
            # self.Couple_Layers.add_module('N1', BatchNormFlow(M, N))
            self.Couple_Layers.add_module('L1', CoupleLayer(M, N, fn, bn, key=0, reverse=False))
            self.Couple_Layers.add_module('L2', CoupleLayer(M, N, fn, bn, key=0, reverse=True))
            self.Couple_Layers.add_module('N2', LayerNormFlow(M, N))

    def forward(self, X, invert=False):
        input_M = X[:, :self.M]
        input_N = X[:, -self.N:]
        if not invert:
            for CP in self.Couple_Layers:
                input_M, input_N = CP(input_M, input_N, invert=False)
        else:
            for CP in self.Couple_Layers[::-1]:
                input_M, input_N = CP(input_M, input_N, invert=True)
        out_M, out_N = input_M, input_N
        out = torch.cat((out_M, out_N), dim=-1)

        return out


if __name__ == '__main__':
    model = RealNVP(2, 2, 1, [64, 128], [32])
    i = torch.tensor([[8, 12, 9, 5],
                      [2, 8, 13, 17],
                      [4, -4, 8, -12]], dtype=torch.float32)
    o = model(i, invert=False)
    print(o)
    print(model(o, invert=True))
    # model = CoupleLayer(2, 2, 512, reverse=False)
    # i1 = torch.tensor([[1, 2]], dtype=torch.float32)
    # i2 = torch.tensor([[3, 4]], dtype=torch.float32)
