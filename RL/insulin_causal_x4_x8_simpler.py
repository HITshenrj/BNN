import os

import numpy as np
import itertools
import copy
import pandas as pd
import torch

N = 13



def _swap_rows_and_cols(arr_original, permutation):
    if not isinstance(permutation, list):
        permutation = list(permutation)
    arr = arr_original.copy()
    arr[:] = arr[permutation]
    arr[:, :] = arr[:, permutation]
    return arr


def get_permuted_adj_mats(adj_list):
    """
    Returns adjacency matrices which are valid permutations, meaning that
    the root node (index = 4) does not have any parents.
    :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
    :return perms: list of adjacency matrices
    """
    adj_mat = np.zeros((N, N))
    adj_triu_list = np.triu_indices(N, 1)
    adj_mat[adj_triu_list] = adj_list
    perms = set()

    for perm in itertools.permutations(np.arange(N), N):
        permed = _swap_rows_and_cols(adj_mat, perm)
        if not any(permed[N - 1]):
            perms.add(tuple(permed.reshape(-1)))

    return perms



def getParams(PATIENT_PARA_FILE, name):
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    params = patient_params.loc[patient_params.Name == name].squeeze()
    return params

def rerankVector(state):
    return np.hstack((state[0], state[1], state[4], state[7], state[8], state[10], state[11], state[12], state[2], state[3], state[5], state[6], state[9], state[13]))

class MLP(torch.nn.Module):
    def __init__(self, num):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, num)
        self.layer2 = torch.nn.Linear(num, num)
        self.layer3 = torch.nn.Linear(num, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)

        x = self.layer2(x)
        x = torch.nn.functional.relu(x)

        x = self.layer3(x)

        return x
class RecNN(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        # x [batch_size, seq_len, input_size]
        output, hn = self.rnn(x)
        # print("output shape: {}".format(output.shape))
        # out [seq_len, batch_size, hidden_size]
        x = output.reshape(-1, self.hidden_size)

        # print("after change shape: {}".format(x.shape))
        x = self.linear(x)

        # print("after linear shape: {}".format(x.shape))

        return x
class SimGlucose():
    EAT_RATE = 5  # g/min CHO
    DETA_TIME = 1  # min

    def __init__(self, params, vertex, last_vertex, init_state=None): # flag_list
        self.params = params
        self.init_state = init_state
        self.flag_list = [1] * 13
        self.vertex = vertex
        self.last_vertex = last_vertex
        self.reset()


    def step(self, intervene, carb):
        CHO = self._announce_meal(carb)
        # print(ins, carb, CHO)
        # Detect eating or not and update last digestion amount
        if CHO > 0 and self.last_CHO <= 0:
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = 0
            self.is_eating = True

        if self.is_eating:
            self._last_foodtaken += CHO   # g

        # Detect eating ended
        if CHO <= 0 and self.last_CHO > 0:
            self.is_eating = False

        self.last_CHO = CHO
        # self.last_ins = ins
        self.last_state = self.state
        self.last_all_state = copy.deepcopy(self.state)
        self.state = self.model(self.state, CHO, intervene, self.params, self._last_Qsto, self._last_foodtaken)
        # print(self.state)
        self.obe_state = self.obersevation()
        # print(self.obe_state)
        self.last_state = self.obe_last_state()

        # print('next state::', self.state)

    def model(self, x, CHO, intervene, params, last_Qsto, last_foodtaken):
        dxdt = np.zeros(13)

        d = CHO * 1000  # g -> mg
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d  # delta_t = 1

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (np.tanh(
                aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - \
                  params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * \
                  x[10] + params.ka2 * x[11]  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = intervene - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glcuose
        dxdt[12] = (-params.ksc * x[12] + params.ksc * x[3])
        dxdt[12] = (x[12] >= 0) * dxdt[12]
        for i in range(13):
            x[i] = x[i] + dxdt[i]
            #x[i] *= self.flag_list[i]
        # print(dxdt[6], x[6])
        return x

    def obersevation(self):
        obe = []
        for i in range(13):
            if self.flag_list[i] == 1:
                obe.append(self.state[i])
        return obe

    def obe_last_state(self):
        obe_last = []
        for i in range(13):
            if i in self.last_vertex:
                j = self.state[i]
                obe_last.append(j)
        return obe_last


    def _announce_meal(self, meal):
        '''
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        '''
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        '''
        Reset the patient state to default intial state
        '''
        self.state = copy.deepcopy(self.init_state)
        self.last_all_state = copy.deepcopy(self.init_state)
        self.obe_state = self.obersevation()
        self.last_state = self.obe_last_state()

        self._last_Qsto = self.state[0] + self.state[1]
        self._last_foodtaken = 0

        self.last_CHO = 0
        self.last_ins = 0

        self.is_eating = False
        self.planned_meal = 0

class CausalGraph:
    def __init__(self,
                 train=True,
                 permute=True,
                 vertex=[],
                 last_vertex=[12],
                 args=None):
        """
        Create the causal graph structure.
        :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
        """
        # if adj_list is None:
        #     adj_list = _get_random_adj_list(train)
        print(vertex)
        print(last_vertex)
        self.vertex = vertex
        self.last_vertex = last_vertex
        self.adj_mat = self.handle(self.vertex, self.last_vertex)
        self.basal = 0.01393558889998341
        self.params = getParams('vpatient_params.csv', args.patient_kind+'#'+args.patient_id)
        self.init_state = []
        self.len_obe = 13
        for item in self.params[2:15]:
            self.init_state.append(item)

        self.simulator = SimGlucose(params=self.params, init_state=self.init_state, vertex=self.vertex,
                                    last_vertex=self.last_vertex) #, flag_list=self.flag_list)

        self.time = 0
        self.meal = args.carbon 
        self.CR = 16 
        self.CF = 42.65337551
        self.target = 140
        self.reset_graph()

    def handle(self, vertex, last_vertex):
        adj_mat = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
             [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
             [0, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],  # 3
             [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # 4
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0],  # 5
             [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # 6
             [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # 7
             [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # 8
             [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # 9
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # 11
             [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 12
        )
        for i in vertex:
            for j in range(13):
                adj_mat[i][j] = 0
        for i in last_vertex:
            for j in range(13):
                adj_mat[j][i] = 0
        return adj_mat

    def reset_graph(self):
        carb = 0
        state = self.init_state
        # first = []
        # for i in range(13):
        #     if i in self.vertex:
        #         first.append(state[i])
        self.simulator.reset()



    def intervene(self, action):
        intervene = action
        carb = 0

       
        if self.time in [ 360, 660, 1080, 1800, 2100, 2520, 3240, 3540, 3960]:
            carb = self.meal

        self.simulator.step(intervene, carb)
        self.time += 1
        return carb

    def sample_all(self):
        return self.simulator.obe_state, self.simulator.last_state

    def get_last_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.simulator.last_all_state[node_idx]

    def get_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        # print("get_value:", self.simulator.state[node_idx])
        return self.simulator.state[node_idx]



