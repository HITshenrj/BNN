import os
from typing import List, Optional, Union
from buffer import ReplayBuffer
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from nets import create_mlp
from utils import layer_init
from probabilistic_ensemble import ProbabilisticEnsemble
from utils import LN
LOG_STD_MAX = 3.5
LOG_STD_MIN = -5


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, action_space, net_arch=[8, 8]):
        super(Policy, self).__init__()
        self.latent_pi = create_mlp(
            input_dim, -1, net_arch)  # 2 layers input->32
        self.mean = nn.Linear(net_arch[-1], output_dim)
        self.logstd = nn.Linear(net_arch[-1], output_dim)
        self.action_low = th.FloatTensor(action_space.low)
        self.action_high = th.FloatTensor(action_space.high)
        th.nn.init.orthogonal(self.latent_pi[0].weight, gain=1)
        th.nn.init.orthogonal(self.latent_pi[2].weight, gain=1)
        th.nn.init.orthogonal(self.mean.weight, gain=1)
        th.nn.init.orthogonal(self.logstd.weight, gain=1)
        self.apply(layer_init)

    def action_dist(self, obs):
        h = self.latent_pi(obs)  # [256, 256]
        mean = self.mean(h)
        log_std = self.logstd(h)
        # MIN <= log_std <= MAX
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def forward(self, obs, deterministic=False):
        mean, log_std = self.action_dist(obs)
        # print(mean, log_std)
        if deterministic:
            return th.tanh(mean)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action = th.tanh(x_t)
        action = th.sigmoid(x_t)
        # action = th.relu(x_t)
        # print('action:', action, 'mean:', mean, 'log_std', log_std)
        # action = (action + 1) / 2 * 10
        return action

    def action_log_prob(self, obs):
        mean, log_std = self.action_dist(obs)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = th.tanh(x_t)
        # action = (action + 1) / 2 * 50
        log_prob = normal.log_prob(x_t).sum(dim=1)
        log_prob -= th.log((1 - action.pow(2)) + 1e-6).sum(dim=1)
        return action, log_prob

    def to(self, device):
        self.action_low = self.action_low.to(device)
        self.action_high = self.action_high.to(device)
        return super(Policy, self).to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, net_arch=[8, 8]):
        super(SoftQNetwork, self).__init__()
        self.net = create_mlp(input_dim, 1, net_arch)
        self.apply(layer_init)

    def forward(self, input):
        q_value = self.net(input)
        return q_value


class SAC:
    # TODO:
    # scale action
    # load save
    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 tau: float = 0.005,
                 buffer_size: int = 1e6,
                 alpha: Union[float, str] = 'auto',
                 net_arch: List = [8, 8],
                 batch_size: int = 256,
                 num_q_nets: int = 2,
                 m_sample: int = 2,  # None == SAC, 2 == REDQ
                 learning_starts: int = 1500,
                 gradient_updates: int = 1,
                 gamma: float = 0.99,
                 mbpo: bool = True,
                 dynamics_rollout_len: int = 1,
                 rollout_dynamics_starts: int = 5000,
                 real_ratio: float = 0.05,
                 project_name: str = 'sac',
                 experiment_name: Optional[str] = None,
                 log: bool = True,
                 wandb: bool = False,
                 device: Union[th.device, str] = 'auto',
                 inn_model=None):

        self.env = env
        # gym:12 our:15
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]  # gym:3  our:1
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets
        self.m_sample = m_sample
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.device = th.device('cuda' if th.cuda.is_available(
        ) else 'cpu') if device == 'auto' else device
        self.replay_buffer = ReplayBuffer(
            self.observation_dim, self.action_dim, max_size=buffer_size)

        self.q_nets = [SoftQNetwork(self.observation_dim+self.action_dim,
                                    net_arch=net_arch).to(self.device) for _ in range(num_q_nets)]
        self.target_q_nets = [SoftQNetwork(
            self.observation_dim+self.action_dim, net_arch=net_arch).to(self.device) for _ in range(num_q_nets)]
        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
            for param in target_q_net.parameters():
                param.requires_grad = False
        self.policy = Policy(self.observation_dim, self.action_dim,
                             self.env.action_space, net_arch=net_arch).to(self.device)
        self.target_entropy = - \
            th.prod(th.Tensor(self.env.action_space.shape)).item()
        # alpha
        if alpha == 'auto':
            self.log_alpha = th.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = optim.Adam(
                [self.log_alpha], lr=self.learning_rate)
        else:
            self.alpha_optim = None
            self.alpha = alpha
        q_net_params = []
        for q_net in self.q_nets:
            q_net_params += list(q_net.parameters())
        self.q_optim = optim.Adam(q_net_params, lr=self.learning_rate)
        # self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=self.learning_rate)
        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=1e-4)
        self.mbpo = mbpo
        if self.mbpo:
            self.dynamics = ProbabilisticEnsemble(input_dim=self.observation_dim + self.action_dim,
                                                  output_dim=self.observation_dim + 1,
                                                  device=self.device)
            #  MB_buffer
            self.dynamics_buffer = ReplayBuffer(self.observation_dim,
                                                self.action_dim,
                                                max_size=400000)
        self.dynamics_rollout_len = dynamics_rollout_len
        self.rollout_dynamics_starts = rollout_dynamics_starts
        self.real_ratio = real_ratio

        self.experiment_name = experiment_name if experiment_name is not None else f"sac_{int(time.time())}"
        self.log = log
        if self.log:
            self.writer = SummaryWriter("runs/{}".format(self.experiment_name))
        self.inn_model = inn_model
        self.state_bias = th.tensor([0.001, 0.01, 0.001, 0.1, 0.001,
                                     0.001, 0.001, 0.001, 0.1, 0.1, 0.1, 0.001]).to(device=self.device)
        self.list_trans = th.tensor([0, 10, 1, 5, 11, 2, 6, 7, 3, 4, 12, 8]).to(device=self.device)
        self.inn_criterion = nn.MSELoss()

    def get_config(self):
        return {'env_id': self.env.unwrapped.spec.id,
                'learning_rate': self.learning_rate,
                'num_q_nets': self.num_q_nets,
                'batch_size': self.batch_size,
                'tau': self.tau,
                'gamma': self.gamma,
                'net_arch': self.net_arch,
                'gradient_updates': self.gradient_updates,
                'm_sample': self.m_sample,
                'buffer_size': self.buffer_size,
                'learning_starts': self.learning_starts,
                'mbpo': self.mbpo,
                'dynamics_rollout_len': self.dynamics_rollout_len}

    def save(self, save_replay_buffer=True):
        save_dir = 'weights/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {'policy_state_dict': self.policy.state_dict(),
                        'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                        'log_alpha': self.log_alpha,
                        'alpha_optimizer_state_dict': self.alpha_optim.state_dict()}
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            saved_params['q_net_'+str(i)+'_state_dict'] = q_net.state_dict()
            saved_params['target_q_net_' +
                         str(i)+'_state_dict'] = target_q_net.state_dict()
        saved_params['q_nets_optimizer_state_dict'] = self.q_optim.state_dict()

        if save_replay_buffer:
            saved_params['replay_buffer'] = self.replay_buffer
            th.save(self.replay_buffer, save_dir +
                    '/' + self.experiment_name + '.txt')

        th.save(saved_params, save_dir + "/" + self.experiment_name + '.tar')

    # def load(self, path, load_replay_buffer=True):
    def load(self, load_replay_buffer=True):
        #path = 'weights/' + 'sac_1500_3time_20W.tar'
        path = 'weights/' + 'redq_1500_3time_20W.tar'
        #path = 'weights/' + 'req_adol5_12carb_3time.tar'
        #path = 'weights/' + 'req_adult10_12carb_bufferload_9.18.tar'
        print(path)
        params = th.load(path)
        self.policy.load_state_dict(params['policy_state_dict'])
        self.policy_optim.load_state_dict(
            params['policy_optimizer_state_dict'])
        self.log_alpha = params['log_alpha']
        self.alpha_optim.load_state_dict(params['alpha_optimizer_state_dict'])
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(params['q_net_'+str(i)+'_state_dict'])
            target_q_net.load_state_dict(
                params['target_q_net_'+str(i)+'_state_dict'])
        self.q_optim.load_state_dict(params['q_nets_optimizer_state_dict'])
        if load_replay_buffer and 'replay_buffer' in params:
            self.replay_buffer = params['replay_buffer']

    def load_buffer_only(self, load_replay_buffer=True):
        path = 'weights/' + 'sac_1500_3time_20W.tar'
        #path = 'weights/' + 'req_adole6_12carb_3time.tar'
        #path = 'weights/' + 'req_ado6_12carb_buffer_nointer.tar'
       # path = 'weights/' + 'req_adult1_12carb_bufferload.tar'
        print(path)
        params = th.load(path)
        if load_replay_buffer and 'replay_buffer' in params:
            self.replay_buffer = params['replay_buffer']

    def sample_batch_experiences(self):
        if not self.mbpo or self.num_timesteps < self.rollout_dynamics_starts:
            return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
        else:
            num_real_samples = int(self.batch_size * 0.05)
            s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.replay_buffer.sample(
                num_real_samples, to_tensor=True, device=self.device)
            m_obs, m_actions, m_rewards, m_next_obs, m_dones = self.dynamics_buffer.sample(
                self.batch_size-num_real_samples, to_tensor=True, device=self.device)
            experience_tuples = (th.cat([s_obs, m_obs], dim=0),
                                 th.cat([s_actions, m_actions], dim=0),
                                 th.cat([s_rewards, m_rewards], dim=0),
                                 th.cat([s_next_obs, m_next_obs], dim=0),
                                 th.cat([s_dones, m_dones], dim=0))
            return experience_tuples

    @property
    def dynamics_train_freq(self):
        if self.num_timesteps < 100000:
            return 250
        else:
            return 1000

    def inn_action(self, s_obs, s_next_obs):
        s_obs_trans = th.index_select(s_obs, 1, self.list_trans)
        s_next_obs_trans = th.index_select(s_next_obs, 1, self.list_trans)
        s_obs_trans = s_obs_trans*self.state_bias
        s_next_obs_trans = s_next_obs_trans*self.state_bias
        LN_St, _, _ = LN(s_obs_trans)
        LN_St1, _, _ = LN(s_next_obs_trans)
        pre_In = self.inn_model.back_ward(LN_St, LN_St1)
        return pre_In

    def train(self):
        for _ in range(self.gradient_updates):
            s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.sample_batch_experiences()

            with th.no_grad():
                inn_action = self.inn_action(s_obs, s_next_obs)
                next_actions, log_probs = self.policy.action_log_prob(
                    s_next_obs)
                q_input = th.cat([s_next_obs, next_actions], dim=1)
                if self.m_sample is not None:  # REDQ sampling
                    q_targets = th.cat([q_target(q_input) for q_target in np.random.choice(
                        self.target_q_nets, self.m_sample, replace=False)], dim=1)
                else:
                    q_targets = th.cat([q_target(q_input)
                                       for q_target in self.target_q_nets], dim=1)

                target_q, _ = th.min(q_targets, dim=1, keepdim=True)
                target_q -= self.alpha * log_probs.reshape(-1, 1)
                target_q = s_rewards + (1 - s_dones) * self.gamma * target_q

            sa = th.cat([s_obs, s_actions], dim=1)
            q_values = [q_net(sa) for q_net in self.q_nets]
            critic_loss = (1/self.num_q_nets) * \
                sum([F.mse_loss(q_value, target_q) for q_value in q_values])

            self.q_optim.zero_grad()
            critic_loss.backward()
            # print(critic_loss)
            self.q_optim.step()

            # Polyak update
            for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
                for param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

        # Policy update
        actions, log_pi = self.policy.action_log_prob(s_obs)
        sa = th.cat([s_obs, actions], dim=1)
        q_values_pi = th.cat([q_net(sa) for q_net in self.q_nets], dim=1)
        # if self.m_sample is not None:
        #     min_q_value_pi = th.mean(q_values_pi, dim=1, keepdim=True)
        # else:
        min_q_value_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        # print(actions.shape)
        # print(inn_action.shape)
        policy_loss = (self.alpha * log_pi - min_q_value_pi).mean() + \
            0.001*self.inn_criterion(actions, inn_action[:,0])
        # print("Policy_loss:{}".format(policy_loss))

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Automatic temperature learning
        if self.alpha_optim is not None:
            alpha_loss = (-self.log_alpha *
                          (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # Log losses
        if self.log and self.num_timesteps % 100 == 0:
            print("==========self.num_timesteps:{}================".format(
                self.num_timesteps))
            self.writer.add_scalar("losses/critic_loss",
                                   critic_loss.item(), self.num_timesteps)
            self.writer.add_scalar("losses/policy_loss",
                                   policy_loss.item(), self.num_timesteps)
            self.writer.add_scalar(
                "losses/alpha", self.alpha, self.num_timesteps)
            if self.alpha_optim is not None:
                self.writer.add_scalar(
                    "losses/alpha_loss", alpha_loss.item(), self.num_timesteps)
            print("losses/critic_loss", critic_loss.item(), self.num_timesteps)
            print("losses/policy_loss", policy_loss.item(), self.num_timesteps)
            print("losses/alpha", self.alpha, self.num_timesteps)
            if self.alpha_optim is not None:
                print("losses/alpha_loss", alpha_loss.item(), self.num_timesteps)

    def learn(self, total_timesteps):
        episode_reward = 0.0,
        num_episodes = 0
        obs, done = self.env.reset(), False
        self.num_timesteps = 0

        Action = []
        Now_glu = []
        Reward = []
        Now_step = 0
        # print(self.env.action_space)
        for step in range(1, total_timesteps+1):
            self.num_timesteps += 1
            if step < self.learning_starts:
                # if step < 5:
                action = self.env.action_space.sample()
            else:
                with th.no_grad():
                    action = self.policy(th.tensor(obs).float().to(
                        self.device)).detach().cpu().numpy()
                    action = action * 15
            next_obs, reward, done, info = self.env.step(
                action)  # 3.13 env return obs
            Action.append(action[0])
            Now_glu.append(next_obs[0][-3])
            Reward.append(reward[0])

            terminal = done if 'TimeLimit.truncated' not in info else not info[
                'TimeLimit.truncated']
            self.replay_buffer.add(obs, action, reward,
                                   next_obs, terminal)

            if step >= self.learning_starts:
                if self.mbpo:
                    if self.num_timesteps % self.dynamics_train_freq == 0:
                        m_obs, m_actions, m_rewards, m_next_obs, m_dones = self.replay_buffer.get_all_data()
                        X = np.hstack((m_obs, m_actions))
                        Y = np.hstack((m_rewards, m_next_obs - m_obs))
                        mean_holdout_loss = self.dynamics.train_ensemble(X, Y)
                        print("step:{}".format(step))
                        self.writer.add_scalar(
                            "dynamics/mean_holdout_loss", mean_holdout_loss, self.num_timesteps)
                        print("dynamics/mean_holdout_loss",
                              mean_holdout_loss, self.num_timesteps)
                    if self.num_timesteps >= self.rollout_dynamics_starts and self.num_timesteps % 250 == 0:
                        # rollout
                        self.rollout_dynamics()
                self.train()

            episode_reward += reward
            # print(episode_reward)
            if done:
                if num_episodes >= 0:

                    print("======step:{}======".format(step))
                    print("num_episodes:", num_episodes, "step:",
                          step, "episode_reward:", episode_reward)
                    Now_glu, Action, Reward = [], [], []

                obs, done = self.env.reset(), False

                num_episodes += 1
                print("Episode: {} Step: {}, Ep. Reward: {}".format(
                    num_episodes, step, episode_reward))
                print("metrics/episode_reward",
                      episode_reward, self.num_timesteps)

                episode_reward = 0.0
            else:
                obs = next_obs

        if self.log:
            self.writer.close()
        self.env.close()
