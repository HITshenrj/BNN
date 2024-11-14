from Net.OWN import OWNNorm
from Net.OWN_linear import OWNLinear
from Net.couple_layers import RealNVP
from casual_tree.Graph import Graph
from casual_tree.Tree import Tree
from utils import compute_num, binary_split
from baseline.MLP import usual_MLP as MLP

from abc import ABC
import torch
import torch.nn as nn


class Graph2Model(nn.Module, ABC):
    def __init__(self, adj_matrix, Ux: list,layers,fn,bn,hidden_num,weight):
        """adjacency matrix convert to INN"""
        super(Graph2Model, self).__init__()
        self.adj_matrix = adj_matrix
        self.Graph = Graph(self.adj_matrix)
        self.Ux = Ux
        self.Ux.sort()
        self.Casual_Tree = Tree(self.Ux, self.Graph)

        self.layers_inf = self.Casual_Tree.compute_each_layers()

        """
         rules:
            1. reverse construction
            2. as far as possible binary split
        """

        # compute splitting
        self.binary_split_list = [0] + binary_split(self.layers_inf)

        # compute each sub_model_layers
        self.layers = []
        for i in range(len(self.binary_split_list)):
            try:
                self.layers.append(self.layers_inf[self.binary_split_list[i]:self.binary_split_list[i + 1]])
            except IndexError:
                self.layers.append(self.layers_inf[self.binary_split_list[i]:])

        self.layers_width = [compute_num(_) for _ in self.layers] 

        # model
        # OWNs and RealNVPs
        self.own_ = OWNNorm(norm_groups=1)
        self.Ws = nn.ModuleList()
        self.CPs = nn.ModuleList()

        up = 0
        for i in range(len(self.layers_width)):
            # After each OWN, a RealNVP is added
            try:
                up += self.layers_width[i]
                down = self.layers_width[i + 1]
                if i != len(self.layers_width) - 2:
                    out = up + down
                    cp_up = up
                else:
                    out = up + down - len(self.Ux)
                    cp_up = up - len(self.Ux)
                W = OWNLinear(up, out)

                CP = RealNVP(cp_up, down, layers=layers,fn=fn,bn=bn)  #layers = 4
                self.Ws.append(W)
                self.CPs.append(CP)
            except IndexError:
                break

        # tips to normalize input
        print("Dimension name of all input:")
        s = ""
        for i, _ in enumerate(self.layers_inf):
            if i == 0:
                for ii in _:
                    s += 'U' + str(ii) + ' '
            else:
                for ii in _:
                    s += str(ii) + ' '
        print(s)
        print("Dimension name of all output:")
        s = ""
        num = 0
        for _ in self.layers_inf[1:]:
            for ii in _:
                s += str(ii) + ' '
                num += 1
        print(s)
        # self.expand = nn.Parameter(torch.tensor([weight] * 12).to(dtype=torch.float32))
        # self.W_forward = MLP(num, num,hidden_num,1)
        self.W_backward = MLP(self.layers_width[0], self.layers_width[0],hidden_num,0)
        # print(num,self.layers_width[0])

    def forward(self, X, own=True):
        """forward process"""
        split_X = []
        cur_idx = 0
        for i in self.layers_width:
            split_X.append(X[:, cur_idx:cur_idx + i])
            cur_idx += i

        control = split_X[0]

        for i in range(len(self.layers_width) - 1):
            be_controlled = split_X[i + 1]
            out = self.Ws[i](control)

            up = out[:, :-be_controlled.size(-1)]

            control_add = out[:, -be_controlled.size(-1):]

            down = be_controlled + control_add
            CP_in = torch.cat((up, down), dim=-1)

            control = self.CPs[i](CP_in, invert=False)
        # print(control)
        # print(control * self.expand)
        # print(self.expand)
        # out = self.W_forward(control * self.expand)
        out = control
        return out

    def back_ward(self, X, Y, Origin_X):
        """backward process"""

        split_X = []
        cur_idx = 0
        for i in self.layers_width:
            split_X.append(X[:, cur_idx:cur_idx + i])
            cur_idx += i

        CP_out = Y
        for i in range(len(self.layers_width) - 2, -1, -1):
            CP_in = self.CPs[i](CP_out, invert=True)
            cur_x = split_X[i + 1]
            up = CP_in[:, :-cur_x.size(-1)]
            down = CP_in[:, -cur_x.size(-1):]
            W_out = torch.cat((up, down - cur_x), dim=-1)
            CP_out = self.Ws[i].back_ward(W_out)

        CP_out = self.W_backward(CP_out)
        pred_In = CP_out
        true_In = Origin_X[:,:2]

        return pred_In, true_In


if __name__ == '__main__':
    e01 = 0.46122
    e02 = 0.001536032069546028
    e12 = 0.03330367437547978
    e23 = 0.0007833659108679641
    e3_12 = 0.0766
    e43 = 0.087114
    e56 = 0.5063563180709586
    e57 = 0.08446071467598117
    e64 = 0.2951511925794614
    e78 = 0.0046374
    e83 = 0.00469
    e10_5 = 0.0019
    e10_11 = 0.0152
    e11_5 = 0.0078

    #                     0  1  2  3  4  5  6  7  8  9  10 11 12
    model = Graph2Model(
                        [[0, e01, e02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                         [0, 0, e12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                         [0, 0, 0, e23, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e3_12],  # 3
                         [0, 0, 0, e43, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                         [0, 0, 0, 0, 0, 0, e56, e57, 0, 0, 0, 0, 0],  # 5
                         [0, 0, 0, 0, e64, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                         [0, 0, 0, 0, 0, 0, 0, 0, e78, 0, 0, 0, 0],  # 7
                         [0, 0, 0, e83, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                         [0, 0, 0, 0, 0, e10_5, 0, 0, 0, 0, 0, e10_11, 0],  # 10
                         [0, 0, 0, 0, 0, e11_5, 0, 0, 0, 0, 0, 0, 0],  # 11
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # 12
                        [0, 10])

    input = torch.tensor([[10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.float32)
    print(model(input))

    print(model.back_ward(input, model(input)))
