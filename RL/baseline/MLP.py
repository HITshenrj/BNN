from casual_tree.Graph import Graph
from casual_tree.Tree import Tree
from utils import compute_num, binary_split

from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module, ABC):
    def __init__(self, adj_matrix, Ux: list):
        super(MLP, self).__init__()
        self.adj_matrix = adj_matrix
        self.Graph = Graph(self.adj_matrix)
        self.Ux = Ux
        self.Ux.sort()
        self.Casual_Tree = Tree(self.Ux, self.Graph)

        self.layers_inf = self.Casual_Tree.compute_each_layers()
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
        input_num = 0
        # tips to normalize input
        print("Dimension name of all input:")
        s = ""
        for i, _ in enumerate(self.layers_inf):
            if i == 0:
                for ii in _:
                    s += 'U' + str(ii) + ' '
                    input_num += 1
            else:
                for ii in _:
                    s += str(ii) + ' '
                    input_num += 1
        for _ in self.layers_inf[1:]:
            for ii in _:
                s += str(ii) + ' '
                input_num += 1
        print(s)
        output_num = 0
        print("Dimension name of all output:")
        s = ""
        for i, _ in enumerate(self.layers_inf):
            if i == 0:
                for ii in _:
                    s += 'U' + str(ii) + ' '
                    output_num += 1

            else:
                for ii in _:
                    s += str(ii) + ' '
                    output_num += 1
        print(s)

        self.hidden1 = nn.Linear(in_features=input_num, out_features=512, bias=True)
        self.hidden2 = nn.Linear(512, 1024)  # 100*50
        self.predict = nn.Linear(1024, output_num)  # 50*1  预测只有一个 房价

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        output = self.predict(x)
        return output


class forward_mlp(nn.Module, ABC):
    def __init__(self, input_num,output_num,hidden_num,k):
        super(forward_mlp, self).__init__()
        self.hidden1 = nn.Linear(in_features=input_num, out_features=hidden_num[k], bias=True)
        self.predict = nn.Linear(hidden_num[k], output_num)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        output = self.predict(x)
        # print("111")
        return output

class usual_MLP(nn.Module, ABC):
    def __init__(self, input_num,output_num,hidden_num,k):
        super(usual_MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=input_num, out_features=hidden_num[k], bias=True)
        self.hidden2 = nn.Linear(in_features=hidden_num[k], out_features=hidden_num[k], bias=True)
        self.predict = nn.Linear(hidden_num[k], output_num)
    def forward(self, x):
        x = F.tanh(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        output = self.predict(x)
        # print("111")
        return output
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

    #              0  1  2  3  4  5  6  7  8  9  10 11  12
    adj_matrix = [[0, e01, e02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
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
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 12

    Ux = [0, 10]

    model = MLP(adj_matrix, Ux)
    input = torch.tensor([[0,10,0,10,1,5,11,2,6,7,3,4,12,8,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=torch.float32)
    print(model(input))

