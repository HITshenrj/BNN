from collections import defaultdict
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import copy

from insulin_causal_x4_x8_simpler import CausalGraph
import torch
import torch.nn.functional as F

import operator
from functools import reduce
import os

from utils import LN
import torch.nn as nn


def one_hot(length, idx):
    one_hot = np.zeros(length)
    one_hot[idx] = 1
    return one_hot


class CBNEnv(Env):
    def __init__(self,
                 agent_type='default',
                 info_phase_length=4320,
                 # info_phase_length=1440,
                 #action_range=[-np.inf, np.inf],
                 action_range=[0, 50],
                 vertex=[5],
                 reward_scale=[1.0, 0.01],
                 list_last_vertex=[
                     {  
                     }
                 ],
                 args=None,
                 inn_model=None
                 ):
        """
        Create a stable_baselines-compatible environment to train policies on
        """
        # vg = 1.886280201
        self.logger = None
        self.log_data = defaultdict(int)
        self.agent_type = agent_type
        self.ep_rew = 0
        self.reward_scale = reward_scale
        self.vertex = vertex
        self.list_last_vertex = list_last_vertex
        self.reward_env = []
        self.reward_goal = []
        self.reward_state = []

        self.last_vertex = [12]
        self.goal = [[70*1.886280201, 180*1.886280201]]

        print(self.vertex)
        print(self.last_vertex)

        self.state = TrainEnvState(
            self.vertex, self.last_vertex, info_phase_length, args)

        self.action_space = Box(0, 10, (1,), dtype=np.float64)
        self.observation_space = Box(-np.inf, np.inf,
                                     (self.state.graph.len_obe + 2 * len(self.goal),))

        self.reset_low = args.reset_low
        self.reset_high = args.reset_high

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.inn_model = inn_model
        self.inn_eta = args.eta
        self.inn_criterion = nn.MSELoss()
        self.now_state_bias = torch.tensor([1, 1, 0.001, 0.01, 0.001, 0.1, 0.001,
                                        0.001, 0.001, 0.001, 0.1, 0.1, 0.1, 0.001]).to(device=self.device)
        self.next_state_bias = torch.tensor([0.001, 0.01, 0.001, 0.1, 0.001,
                                         0.001, 0.001, 0.001, 0.1, 0.1, 0.1, 0.001]).to(device=self.device)
        self.tot_intr_reward = 0
        self.tot_r = 0

        self.state_list = []
        

    @classmethod
    def create(cls, info_phase_length, vertex, reward_scale, list_last_vertex, action_range, n_env, args, inn_model):
        return DummyVecEnv([lambda: cls(info_phase_length=info_phase_length,
                                        vertex=vertex,
                                        reward_scale=reward_scale,
                                        list_last_vertex=list_last_vertex,
                                        action_range=action_range,
                                        args=args,
                                        inn_model=inn_model)
                            for _ in range(n_env)])

    def reward(self, val, observed_valse, pre_glu):
        if len(val) == 1:
            max_state = torch.from_numpy(np.array(val))
            pre_state = torch.from_numpy(np.array(pre_glu))
        
            score_offset = abs(
                F.relu(self.goal[0][0] - max_state[0]) + F.relu(max_state[0] - self.goal[0][1]))
            score_center_plus = abs(
                min(self.goal[0][1], max(self.goal[0][0], max_state[0])) - (self.goal[0][0] + self.goal[0][1]) / 2)
            score_center_plus = 24 - score_offset - 0.2 * score_center_plus
           
            if abs((max_state[0]-140 * 1.886280201).numpy()) < abs((pre_state-140 * 1.886280201).numpy()):
                score_center_plus = 12 - score_offset - 0.2 * \
                    score_center_plus + 0.15 * score_center_plus
            else:
                score_center_plus = 12 - score_offset - 0.2 * score_center_plus

        elif len(val) == 2:
            max_state = torch.from_numpy(np.array(val))
            score_offset = abs(
                F.relu(self.goal[0][0] - max_state[0]) + F.relu(max_state[0] - self.goal[0][1]))
            score_center_plus = abs(
                min(self.goal[0][1], max(self.goal[0][0], max_state[0])) - (self.goal[0][0] + self.goal[0][1]) / 2)
            score_center_plus = 24 - score_offset - 0.2 * score_center_plus
            score_offset_ = abs(
                F.relu(self.goal[1][0] - max_state[1]) + F.relu(max_state[1] - self.goal[1][1]))
            score_center_plus_ = abs(
                min(self.goal[1][1], max(self.goal[1][0], max_state[1])) - (self.goal[1][0] + self.goal[1][1]) / 2)
            score_center_plus_ = 24 - score_offset_ - 0.2 * score_center_plus_
            score_center_plus += score_center_plus_

        
        if (self.state.info_steps >= 360 and self.state.info_steps <= 420) \
                or (self.state.info_steps >= 660 and self.state.info_steps <= 720) \
                or (self.state.info_steps >= 1080 and self.state.info_steps <= 1140)\
                or (self.state.info_steps >= 1800 and self.state.info_steps <= 1860)\
                or (self.state.info_steps >= 2100 and self.state.info_steps <= 2160)\
                or (self.state.info_steps >= 2520 and self.state.info_steps <= 2580)\
                or (self.state.info_steps >= 3240 and self.state.info_steps <= 3300)\
                or (self.state.info_steps >= 3540 and self.state.info_steps <= 3600)\
                or (self.state.info_steps >= 3960 and self.state.info_steps <= 4020):
            r2 = 0.0
        else:
            temp_reward = 0
            temp_reward = self.vertex[0]
            r2 = - 1.0 * abs(self.state.get_value(temp_reward) -
                             self.state.get_last_value(temp_reward))
        # print("score_center_plus",score_center_plus,"r2:",r2)
        return self.reward_scale[0] * score_center_plus + self.reward_scale[1] * r2
        # return score_center_plus

    def inn_reward(self, ins, carb, pre_observed_vals, observed_vals):
        self.inn_model.eval()
        if isinstance(ins, np.ndarray):
            # print(1)
            now_state_numpy = np.array([[ins[0], carb, pre_observed_vals[0], pre_observed_vals[10], pre_observed_vals[1], pre_observed_vals[5], pre_observed_vals[11], pre_observed_vals[2],pre_observed_vals[6], pre_observed_vals[7], pre_observed_vals[3], pre_observed_vals[4], pre_observed_vals[12], pre_observed_vals[8]]])
        else:
            now_state_numpy = np.array([[ins, carb, pre_observed_vals[0], pre_observed_vals[10], pre_observed_vals[1], pre_observed_vals[5], pre_observed_vals[11], pre_observed_vals[2],pre_observed_vals[6], pre_observed_vals[7], pre_observed_vals[3], pre_observed_vals[4], pre_observed_vals[12], pre_observed_vals[8]]])
        now_state = torch.tensor(now_state_numpy).to(device=self.device)
        now_state=now_state*self.now_state_bias
        
        next_state_numpy = np.array([[observed_vals[0], observed_vals[10], observed_vals[1], observed_vals[5], observed_vals[11], observed_vals[2], observed_vals[6], observed_vals[7], observed_vals[3], observed_vals[4], observed_vals[12], observed_vals[8]]])
        next_state = torch.tensor(next_state_numpy).to(device=self.device)
        next_state=next_state*self.next_state_bias
        LN_at_St, _, _ = LN(now_state)
        LN_St1, mean, var = LN(next_state)

        LN_at_St = LN_at_St.to(device=self.device,dtype=torch.float32)
        # print(LN_at_St)
        next_state = next_state.to(device=self.device,dtype=torch.float32)
        mean = mean.to(device=self.device,dtype=torch.float32)
        var = var.to(device=self.device,dtype=torch.float32)
        
        LN_St1_hat = self.inn_model(LN_at_St)
        forward_loss = self.inn_eta*self.inn_criterion(
            LN_St1_hat * var.sqrt() + mean, next_state).item()
        # print(forward_loss)
        return forward_loss

    def step(self, action):
        info = dict()
        pre_observed_vals, pre_glu = self.state.sample_all()
        pre_glu = pre_glu[0]
        carb = self.state.intervene(action)
        observed_vals, now_glu = self.state.sample_all()

        intr_reward = self.inn_reward(
            action, carb, pre_observed_vals, observed_vals)
        self.tot_intr_reward+=intr_reward
        r = self.reward(now_glu, observed_vals, pre_glu).numpy()+intr_reward
        self.tot_r+=r

        now_glu = now_glu[0]
        self.state_list.append(now_glu)

        reset_low = self.reset_low * 1.886280201
        reset_high = self.reset_high * 1.886280201
        if (self.state.info_steps != self.state.info_phase_length) and (now_glu > reset_low) and (now_glu < reset_high):
            self.ep_rew += r
            done = False
        else:
            info["episode"] = dict()
            info["episode"]["r"] = self.ep_rew
            info["episode"]["l"] = self.state.info_steps
            self.ep_rew = 0.0

            print("======== tot_intr_reward: {} ========".format(self.tot_intr_reward))
            print("======== tot_r: {} ========".format(self.tot_r))
            print("======== state_list: {} ========".format(self.state_list))
            done = True

        # concatenate all data that goes into an observation
        obs_tuple = np.hstack([observed_vals, reduce(operator.add, self.goal)])
        obs = obs_tuple
        new_prev_action = np.array(action)
        self.state.step_state(new_prev_action, np.array([r]), obs)
        return obs, r, done, info

    def log_callback(self):
        for k, v in self.log_data.items():
            self.logger.logkv(k, v)
        self.log_data = defaultdict(int)

    def reset(self):
        self.state.reset()
        observed_vals, now_insulin = self.state.sample_all()
        obs_tuple = np.hstack((observed_vals, reduce(operator.add, self.goal)))
        obs = obs_tuple
        self.tot_intr_reward = 0
        self.tot_r = 0
        self.state_list = []
        self.state_list.append(now_insulin[0])
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=94566):
        np.random.seed(seed)


class EnvState(object):
    def __init__(self, vertex, last_vertex, info_phase_length=50, args=None):
        """Create an object which holds the state of a CBNEnv"""
        self.info_phase_length = info_phase_length
        self.info_steps = None
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.graph = None
        self.vertex = vertex
        self.last_vertex = last_vertex
        self.args = args

        self.reset()

    def step_state(self, new_prev_action, new_prev_reward, new_prev_state):
        self.prev_action = new_prev_action
        self.prev_reward = new_prev_reward
        self.prev_state = copy.deepcopy(new_prev_state)
        if self.info_steps == self.info_phase_length:
            self.info_steps = 0
        else:
            self.info_steps += 1

    def intervene(self, intervene_val):
        return self.graph.intervene(intervene_val)

    def sample_all(self):
        return self.graph.sample_all()

    def get_value(self, node_idx):
        return self.graph.get_value(node_idx)

    def get_last_value(self, node_idx):
        return self.graph.get_last_value(node_idx)

    def get_graph(self):
        raise NotImplementedError()

    def reset(self):
        self.info_steps = 0
        self.prev_action = np.zeros(1)
        self.prev_reward = np.zeros(1)
        self.graph = self.get_graph()
        self.prev_state, _ = self.sample_all()


class TrainEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=True, vertex=self.vertex, last_vertex=self.last_vertex, args=self.args)


class TestEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=False, vertex=self.vertex, last_vertex=self.last_vertex, args=self.args)


class DebugEnvState(EnvState):
    def __init__(self):
        super().__init__()
        self.reward_data = None
