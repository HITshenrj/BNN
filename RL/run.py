import numpy as np
from sac_continuos_action import SAC
from env_do_all_her_2reward import CBNEnv
import argparse
from G2M_model import Graph2Model
import torch


def run(args, adj_matrix, Ux):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    inn_model = Graph2Model(adj_matrix, Ux, args.layers, args.fn,
                            args.bn, args.hidden, args.weight)
    inn_model.load_state_dict(torch.load(
        "ckpt/"+args.load_path, map_location=device))
    inn_model.to(device=device)
    # print(inn_model)

    env = CBNEnv.create(
        info_phase_length=4320,
        action_range=args.action_range,
        vertex=args.vertex,
        reward_scale=args.reward_scale,
        list_last_vertex=[],
        n_env=1,
        args=args,
        inn_model=inn_model
    )
    print('env set')
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))   # 1
    # print("obs_dim: ", obs_dim, "action_dim: ", action_dim)     # obs_dim:  15 action_dim:  1

    agent = SAC(env,
                gradient_updates=20,
                num_q_nets=2,
                m_sample=None,
                buffer_size=int(4e5),
                mbpo=False,
                experiment_name=args.experiment_name,
                log=True,
                wandb=False,
                inn_model=inn_model)
    #print('agent set')
    agent.learn(total_timesteps=150000)
    agent.save()


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--vertex", nargs='+', type=int,
                        help="Now inserve", default=[10])
    parser.add_argument("--action_range", nargs='+', type=int,
                        help=" ", default=[0, np.inf])
    parser.add_argument("--lr", type=float, help=" ",
                        default=float(1e-4))
    parser.add_argument("--reward_scale", nargs='+', type=float,
                        help=" ", default=[1.0, 0.01])  # -reward_scale 1.0 0.01
    parser.add_argument("--n_step", type=int, help="n_step", default=5)
    parser.add_argument("--seed", type=int, help="seed", default=94566)
    parser.add_argument("--experiment_name", type=str,
                        default="sac_adole2_0-450_carb20_new", help="Save dir")
    parser.add_argument("--patient_kind", type=str,
                        default="adolescent", help="patient kind")
    parser.add_argument("--patient_id", type=str,
                        default="006", help="patient id")
    parser.add_argument("--carbon", type=int, default=20, help="carbon")
    parser.add_argument("--reset_low", type=int, default=-30, help="reset_low")
    parser.add_argument("--reset_high", type=int,
                        default=300, help="reset_high")

    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--inn_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--fn', type=int, nargs='+', default=['32', '32'])
    parser.add_argument('--bn', type=int, nargs='+', default=['32', '32'])
    parser.add_argument('--decade_epoch', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--hidden', type=int,
                        nargs='+', default=['64', '0'])
    parser.add_argument('--log_dir', type=str,
                        default='./log/lr_1e-4_f_32_h_64f')
    parser.add_argument('--weight', type=float, default=5.)
    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument("--load_path", type=str,
                        default="006lr_1e-4_f_32_h_64f.pth")
    parser.add_argument("--eta", type=float, default=1e-3)


    args = parser.parse_args()
    args.fn = [int(fn) for fn in args.fn]
    args.bn = [int(bn) for bn in args.bn]
    args.hidden = [int(hidden) for hidden in args.hidden]
    run(args, adj_matrix, Ux)
