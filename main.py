import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from G2M_model import Graph2Model
from config import Glucose_sim_Config


def LN(n):
    data_tensor = copy.deepcopy(n)
    mean = data_tensor.mean(-1)
    var = (data_tensor - mean.reshape(-1, 1)).pow(2).mean(-1)
    data_tensor = (data_tensor - mean.reshape(-1, 1)) / \
        var.sqrt().reshape(-1, 1)
    return data_tensor, mean, var


def generate_data(n, opt):
    # prepare data
    dic = {0: opt.data_path0,
           1: opt.data_path1,
           2: opt.data_path2,
           3: opt.data_path3,
           4: opt.data_path4,
           5: opt.data_path5,
           6: opt.data_path6,
           7: opt.data_path7,
           8: opt.data_path8,
           9: opt.data_path9
           }
    data = np.load(dic[n], allow_pickle=True)
    data_ = []
    bias = [1, 1, 0.001, 0.01, 0.001, 0.1, 0.001,
            0.001, 0.001, 0.001, 0.1, 0.1, 0.1, 0.001]
    for i in range(data.shape[0]):
        if i % 480 == 0 and i >= 960 and i < data.shape[0] - 480:
            d = np.array([data[j, :] for j in range(i, i + 71)]) * bias
            d_ = np.c_[d[:-1, :], d[1:, 2:]]
            data_.append(d_)

    data = np.array([l for s in data_ for l in s])
    np.random.shuffle(data)
    data_tensor = torch.from_numpy(data).to(dtype=torch.float32)


    LN_at_St, _, _ = LN(data_tensor[:, :-12])  # forward input
    LN_St1, _, _, = LN(data_tensor[:, -12:])  # inverse input
    LN_St, mean, var = LN(data_tensor[:, 2:-12])  # LN_St for inverse loss，mean, var for forward loss
    LN_St = torch.cat([torch.zeros([LN_St.shape[0], 2]), LN_St], dim=-1)
    return LN_at_St, LN_St1, LN_St, data_tensor, mean, var


def adjust_learning_rate(optimizer, epoch, start_lr, decade, gamma):
    lr = start_lr * (gamma ** (epoch // decade))
    if lr >= 1e-7:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main(opt, adj_matrix, Ux: list):
    arg_parser = argparse.ArgumentParser()
    # Synthetic data
    # all default
    arg_parser.add_argument('--layers', type=int, default=2)
    arg_parser.add_argument('--lr', type=float, default=0.0001)
    arg_parser.add_argument('--epochs', type=int, default=1000)
    arg_parser.add_argument('--fn', type=int, nargs='+', default=['32', '32'])
    arg_parser.add_argument('--bn', type=int, nargs='+', default=['32', '32'])
    arg_parser.add_argument('--decade_epoch', type=int, default=1000)
    arg_parser.add_argument('--gamma', type=float, default=0.5)
    arg_parser.add_argument('--hidden', type=int,
                            nargs='+', default=['32', '64'])
    arg_parser.add_argument('--log_dir', type=str,
                            default='./log/lr_1e-4_f_32_h_32')
    arg_parser.add_argument('--weight', type=float, default=5.)
    arg_parser.add_argument('--reload', type=bool, default=False)
    args = arg_parser.parse_args()
    args.fn = [int(fn) for fn in args.fn]
    args.bn = [int(bn) for bn in args.bn]
    args.hidden = [int(hidden) for hidden in args.hidden]
    tb_logger = SummaryWriter(args.log_dir)
    save_path = './ckpt/'+args.log_dir[6:]

    model = Graph2Model(adj_matrix, Ux, args.layers, args.fn,
                        args.bn, args.hidden, args.weight)
    epoches = 0
    if args.reload:
        for epoch in range(1, 10):
            path_checkpoint = save_path+'_'+str(epoch*1500)+'.pth'
            if not os.path.exists(path_checkpoint):
                path_checkpoint = save_path+'_'+str((epoch-1)*1500)+'.pth'
                break
        checkpoint = torch.load(
            path_checkpoint, map_location=torch.device('gpu'))
        model.load_state_dict(checkpoint)

    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
    start_lr = args.lr
    # , weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    X0, Y0, Z0, xy0, mean0, var0 = generate_data(0, opt)
    print(X0.shape, Y0.shape, Z0.shape, xy0.shape)
    print(" ")
    X1, Y1, Z1, xy1, mean1, var1 = generate_data(1, opt)
    X2, Y2, Z2, xy2, mean2, var2 = generate_data(2, opt)
    X3, Y3, Z3, xy3, mean3, var3 = generate_data(3, opt)
    X4, Y4, Z4, xy4, mean4, var4 = generate_data(4, opt)
    X5, Y5, Z5, xy5, mean5, var5 = generate_data(5, opt)
    X6, Y6, Z6, xy6, mean6, var6 = generate_data(6, opt)
    X7, Y7, Z7, xy7, mean7, var7 = generate_data(7, opt)
    X8, Y8, Z8, xy8, mean8, var8 = generate_data(8, opt)
    X9, Y9, Z9, xy9, mean9, var9 = generate_data(9, opt)
    # X2, Y2, xy2 = generate_data(2, opt)
    # X3, Y3, xy3 = generate_data(3, opt)
    # X4, Y4, xy4 = generate_data(4, opt)
    # X5, Y5, xy5 = generate_data(5, opt)
    # X6, Y6, xy6 = generate_data(6, opt)
    # X7, Y7, xy7 = generate_data(7, opt)
    # X8, Y8, xy8 = generate_data(8, opt)
    # X9, Y9, xy9 = generate_data(9, opt)
    train_X_f = X0[:60000, :]
    train_X_b = Z0[:60000, :]
    train_Y = Y0[:60000, :]
    mean_weight = mean0.reshape(-1, 1)[:60000]
    var_weight = var0.reshape(-1, 1)[:60000]
    forward = xy0[:60000, -12:]
    backward = xy0[:60000, :-12]

    test_X_f = X0[60000:, :]
    test_X_b = Z0[60000:, :]
    test_Y = Y0[60000:, :]
    test_forward = xy0[60000:, -12:]
    test_backward = xy0[60000:, :-12]
    test_mean_weight = mean0.reshape(-1, 1)[60000:]
    test_var_weight = var0.reshape(-1, 1)[60000:]

    flag = False
    count = 0
    delta_loss = 0.0
    for epoch in range(args.epochs+1):
        adjust_learning_rate(optimizer, epoch, start_lr,
                             args.decade_epoch, args.gamma)
        # print('\n=====epoch {0} ====='.format(epoch + 1))
        model.train()
        epoch_loss = 0.0
        inverse_loss = 0.0

        for step in range(0, len(test_X_f), opt.batch_size):
            # print(torch.matmul(model.alpha1, model.alpha1.T))
            input_X_f = train_X_f[step:step + opt.batch_size]
            input_X_b = train_X_b[step:step + opt.batch_size]
            input_Y = train_Y[step:step + opt.batch_size]
            forward_ans = forward[step:step + opt.batch_size]
            backward_ans = backward[step:step + opt.batch_size]
            mean0 = mean_weight[step:step + opt.batch_size]
            var0 = var_weight[step:step + opt.batch_size]
            if torch.cuda.is_available():
                input_X_f = input_X_f.cuda()
                input_X_b = input_X_b.cuda()
                input_Y = input_Y.cuda()
                forward_ans = forward_ans.cuda()
                backward_ans = backward_ans.cuda()
                mean0 = mean0.cuda()
                var0 = var0.cuda()

            forward_out = model(input_X_f, own=True)
            forward_loss = criterion(
                forward_out * var0.sqrt() + mean0, forward_ans) * opt.alpha
            epoch_loss += forward_loss.item()

            pred_In, true_In = model.back_ward(
                input_X_b, input_Y, backward_ans)
            backward_loss = criterion(true_In, pred_In) * opt.beta
            inverse_loss += criterion(true_In, pred_In).item()

            loss = forward_loss + backward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0 and epoch != 0:
            torch.save(model.state_dict(), save_path+'_'+str(epoch)+'.pth')

        print('\n=====epoch {0} finished=====\nepoch_forward_loss = {1:.6f}\nepoch_inverse_loss = {2:.6f}'
              .format(epoch + 1, epoch_loss, inverse_loss))
        tb_logger.add_scalars(
            './log_exp_2/' + 'loss_train_forward', {'forward_loss': epoch_loss}, epoch)
        tb_logger.add_scalars(
            './log_exp_2/' + 'loss_train_inverse', {'inverse_loss': inverse_loss}, epoch)

        for_loss, invert_loss = invertible_Inference(
            opt, model,
            (test_X_f, test_X_b, test_Y, test_forward, test_backward,
             test_mean_weight, test_var_weight)
        )
        print('=====0forward loss {0:.4f} inverse loss: {1:.4f} ====='.format(
            for_loss, invert_loss))
        tb_logger.add_scalars(
            './log_exp_2/' + 'loss_test_forward_0', {'forward_loss': for_loss}, epoch)
        tb_logger.add_scalars(
            './log_exp_2/' + 'loss_test_inverse_0', {'inverse_loss': invert_loss}, epoch)

        for i in range(1, 10):
            locals()["for_loss" + str(i)], locals()["invert_loss" + str(i)] = invertible_Inference(
                opt, model,
                (
                    locals()["X" + str(i)], locals()["Z" +
                                                     str(i)], locals()["Y" + str(i)],
                    locals()["xy" + str(i)][:, -
                                            12:], locals()["xy" + str(i)][:, :-12],
                    locals()["mean" + str(i)].reshape(-1,
                                                      1), locals()["var" + str(i)].reshape(-1, 1)
                )
            )

            print('====={0:d}forward loss {1:.4f} inverse loss: {2:.4f} ====='.format(i, locals()["for_loss" + str(i)],
                                                                        locals()["invert_loss" + str(i)]))
            tb_logger.add_scalars('./log_exp_2/' + 'loss_test_forward_{}'.format(i),
                                  {'forward_loss': locals()["for_loss" + str(i)]}, epoch)
            tb_logger.add_scalars('./log_exp_2/' + 'loss_test_inverse_{}'.format(i),
                                  {'inverse_loss': locals()["invert_loss" + str(i)]}, epoch)
        # print(1)


def invertible_Inference(opt, model, test_data):
    # print(test_data.shape)
    """counterfactual inference"""
    # model.orthogonality_weight_normalization()
    model.eval()
    # print(model)
    criterion = nn.MSELoss()
    test_X_f, test_X_b, test_Y, test_forward, test_backward, mean_weight, var_weight = test_data
    # print(test_X_f.shape)
    forward_loss = 0.0
    invert_loss = 0.0

    with torch.no_grad():
        for step in range(0, len(test_X_f), opt.batch_size):
            input_X_f = test_X_f[step:step + opt.batch_size]
            input_X_b = test_X_b[step:step + opt.batch_size]
            input_Y = test_Y[step:step + opt.batch_size]
            forward_ans = test_forward[step:step + opt.batch_size]
            backward_ans = test_backward[step:step + opt.batch_size]
            mean0 = mean_weight[step:step + opt.batch_size]
            var0 = var_weight[step:step + opt.batch_size]
            if torch.cuda.is_available():
                input_X_f = input_X_f.cuda()
                input_X_b = input_X_b.cuda()
                input_Y = input_Y.cuda()
                forward_ans = forward_ans.cuda()
                backward_ans = backward_ans.cuda()
                mean0 = mean0.cuda()
                var0 = var0.cuda()
            forward_out = model(input_X_f, own=False)
            pre_In, true_In = model.back_ward(input_X_b, input_Y, backward_ans)
            forward_loss += criterion(forward_out *
                                      var0.sqrt() + mean0, forward_ans).item()
            invert_loss += criterion(true_In, pre_In).item()

    return forward_loss, invert_loss


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
    opt = Glucose_sim_Config()
    main(opt, adj_matrix, Ux)
