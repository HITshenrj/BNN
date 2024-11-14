import numpy as np
import copy
import pandas as pd


class SimGlucose:
    EAT_RATE = 5  # g/min CHO碳水
    DETA_TIME = 1  # min

    def __init__(self, params, init_state=None):
        self.params = params
        self.init_state = init_state
        self.reset()

    def step(self, ins, carb):
        # 将吃一餐转化为实际饮食量
        CHO = self._announce_meal(carb)
        # print(ins, carb, CHO)
        # Detect eating or not and update last digestion amount
        if CHO > 0 and self.last_CHO <= 0:
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = 0
            self.is_eating = True

        if self.is_eating:
            self._last_foodtaken += CHO  # g

        # Detect eating ended
        if CHO <= 0 and self.last_CHO > 0:
            self.is_eating = False

        self.last_CHO = CHO
        self.last_ins = ins
        self.last_state = self.state
        # print('now state::', self.state)

        self.state = self.model(self.state, CHO, ins, self.params, self._last_Qsto, self._last_foodtaken)

    def model(self, x, CHO, ins, params, last_Qsto, last_foodtaken):
        dxdt = np.zeros(13)

        d = CHO * 1000  # g -> mg
        insulin = ins * 6000 / params.BW  # U/min -> pmol/kg/min

        # Glucose in the stomach
        qsto = x[0] + x[1]  # x[0]：Q_sto1(胃中固体葡萄糖质量)；x[1]：Q_sto2(胃中液体葡萄糖质量)
        Dbar = last_Qsto + last_foodtaken  # 上一时刻固液葡萄糖量 + 上一时刻食物摄入

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (np.tanh(
                aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        # stomach liquid
        # print('0-1:', params.kmax)
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        # print('0-2:', abs(params.kmax * kgut))
        # print('1-2:', abs(kgut))
        dxdt[2] = kgut * x[1] - params.kabs * x[2]  # x[2]：Q_gut(肠中葡萄糖质量)

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]  # x[3]：G_p(t)(血浆中葡萄糖质量)
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        # print('2-3:', abs(params.f * params.kabs / params.BW))
        # print('4-3:', abs(params.k2))
        # print('8-3:', abs(params.kp2))
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - \
                  params.k1 * x[3] + params.k2 * x[4]  # x[4]：G_t(t)(慢速平衡组织中葡萄糖质量)
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]  # x[6]：X(t)(葡萄糖利用中胰岛素的参与量)
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        # print('6-4:', abs(Vmt * x[4] / (Kmt + x[4])))
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics

        # print('10-5:', abs(params.ka1))
        # print('11-5', abs(params.ka2))
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * \
                  x[10] + params.ka2 * x[11]  # plus insulin IV injection u[3] if needed
        # x[5]：I_p(t)(血浆中胰岛素质量)；x[9]：I_l(t)(肾脏中胰岛素质量)；x[10]：I_ev(t)(血管外胰岛素质量)；x[11]：？
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        # print('5-7:', abs(params.ki / params.Vi))
        dxdt[7] = -params.ki * (x[7] - It)  # x[7]:I'(t)(中间室胰岛素浓度)

        # print('7-8:', abs(params.ki))
        dxdt[8] = -params.ki * (x[8] - x[7])  # x[8]:XL(t)(延迟的胰岛素信号)

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        # print('10-11:', abs(params.kd))
        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        # print('3-12:', params.ksc)
        dxdt[12] = (-params.ksc * x[12] + params.ksc * x[3])  # x[12]：血管外葡萄糖质量
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        for i in range(13):
            x[i] = x[i] + dxdt[i]

        return x

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        """
        Reset the patient state to default initial state
        """
        self.state = copy.deepcopy(self.init_state)

        self._last_Qsto = self.state[0] + self.state[1]
        self._last_foodtaken = 0

        self.last_CHO = 0
        self.last_ins = 0

        self.is_eating = False
        self.planned_meal = 0


def getParams(PATIENT_PARA_FILE, name):
    patient_params = pd.read_csv(PATIENT_PARA_FILE, engine='python')
    params = patient_params.loc[patient_params.Name == name].squeeze()
    return params


def rerankVector(state):
    return np.hstack((state[0], state[1], state[4], state[7], state[8], state[10], state[11], state[12], state[2],
                      state[3], state[5], state[6], state[9], state[13]))


if __name__ == '__main__':
    import random



    basal = 1.93558889998341  # 每分钟注射的基础胰岛素剂量  #分布

    params = getParams('vpatient_params.csv', 'adolescent#006')
    init_state = []  # 从x0_1到x0_13
    for item in params[2:15]:
        init_state.append(item)
    # init_state_static = [20000.0, 0.00000000e+00, 0.00000000e+00, 2.50621836e+02, 1.76506560e+02, 4.69751776e+00,
    #                      -1.40721735e-10, 9.75540000e+01, 9.75540000e+01, 3.19814917e+00, 5.79512245e+01,
    #                      9.32258828e+01,
    #                      2.50621836e+02]

    data_dic = {}

    simulator = SimGlucose(params=params, init_state=init_state)

    minutes = 24 * 60 * 7 * 50 - 1
    state_history = np.zeros((minutes, 14))
    time = 0
    balus = 13.  + basal  # 大剂量注射量
    # 这里考虑，初始化的时候先运行两天，然后再开始用RL给出的action
    flag = 0
    while time < minutes:  # time 为minute
        ins = basal  # + 0.005 * np.random.randn()
        carb = 0.   #  + 0.1* random.random()
        if time == 0:
            ins = balus
        if time % 480 == 0  and time != 0 and flag == 0:  # 设置一日三餐： 6:00  11:00  18:00
            print("开始进食_早饭")
            ins = balus + 8. * np.random.random()
            carb = 50 + 5 * random.random()
            flag = 1
        if time % 480 == 0  and time != 0 and flag == 1:  # 设置一日三餐： 6:00  11:00  18:00
            print("开始进食_午饭")
            ins = balus + 5. + 8. * np.random.random()
            carb = 80 + 5 * random.random()
            flag = 2
        if time % 480 == 0  and time != 0 and flag == 2:  # 设置一日三餐： 6:00  11:00  18:00
            print("开始进食_晚饭")
            ins = balus + 2. + 8. * np.random.random()
            carb = 60 + 5 * random.random()
            flag = 0
        simulator.step(ins, carb)
        # if time % 2 == 0:
        cur_state = simulator.state
        # state_history[time] = np.concatenate(np.array([ins, carb]), cur_state)
        state_history[time] = np.array(
            [ins, carb, cur_state[0], cur_state[10], cur_state[1], cur_state[5], cur_state[11], cur_state[2],
             cur_state[6],
             cur_state[7], cur_state[3], cur_state[4], cur_state[12], cur_state[8]])
        # print([ins, carb, cur_state[0], cur_state[10], cur_state[1], cur_state[11], cur_state[2], cur_state[5],
        #        cur_state[4], cur_state[8], cur_state[3], cur_state[12]])
        time += 1
    name = ['ins', 'carb', '0', '10', '1', '5', '11', '2', '6', '7', '3', '4', '12', '8']
    for i, _ in enumerate(state_history.var(0)):
        print(name[i]+':', '{0:.4f}'.format(_))
    np.save('./Glucose_sim_data_adolescent006.npy', state_history)
        # np.save('./adolescent#001.npy', state_history)

