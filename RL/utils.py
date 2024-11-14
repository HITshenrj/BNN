import os
import numpy as np
import torch as th
from torch import nn
import torch
import copy

def LN(n):
    data_tensor = copy.deepcopy(n)
    mean = data_tensor.mean(-1)
    var = (data_tensor - mean.reshape(-1, 1)).pow(2).mean(-1)
    data_tensor = (data_tensor - mean.reshape(-1, 1)) / \
        var.sqrt().reshape(-1, 1)
    return data_tensor, mean, var

def MSE(label_Y, pred_Y):
    """
    :param label_Y
    :param pred_Y
    :return:
    """

    batch_size, feature_dim = pred_Y.size()
    # print(pred_Y.size())
    minus = label_Y - pred_Y
    # print(minus)
    loss = torch.mul(minus, minus)
    loss = torch.sum(loss, dim=-1)
    loss /= feature_dim
    loss = torch.sum(loss, dim=0)
    loss /= batch_size

    return loss.cpu().item()

def prepareData(filePath):
    samples_dict = np.load(filePath, allow_pickle=True).item()

    samples_matrix = np.vstack((samples_dict['M1'], samples_dict['M3'], samples_dict['M3'], samples_dict['M4'],
                                samples_dict['M5'], samples_dict['M5'], samples_dict['M7'], samples_dict['M8'],
                                samples_dict['Insulin']))

    samples_train = samples_matrix[:, :]

    return samples_train


def orthogonality_constraint(T):
    """Orthogonality Constraints"""
    return torch.sum(T[0] * T[1], dim=-1)


def compute_num(l):
    """Count the number of elements"""
    total_num = 0
    for _ in l:
        total_num += len(_)
    return total_num


def binary_split(l):
    """Divide the group into two parts as much as possible according to the level"""
    if len(l) == 1:
        return []
    total_num = compute_num(l)
    cur_num = 0
    for i, sub_l in enumerate(l):
        cur_num += len(sub_l)
        if cur_num >= total_num / 2.0:
            if i != len(l) - 1:
                return binary_split(l[:i + 1]) + [i + 1]
            else:
                return [i]

def layer_init(layer, method='xavier', weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if method == "xavier":
            th.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == "orthogonal":
            th.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        th.nn.init.constant_(layer.bias, bias_const)

if __name__ == '__main__':
    print(binary_split([[0, 1], [0, 1], [2, 3, 4, 5], [6, 7, 9, 10], [8], [11]]))
