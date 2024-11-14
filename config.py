class Glucose_sim_Config(object):
    data_path0 = 'Data/Glucose_sim_data_adolescent006_0.npy'
    data_path1 = 'Data/Glucose_sim_data_adolescent006_1.npy'
    data_path2 = 'Data/Glucose_sim_data_adolescent006_2.npy'
    data_path3 = 'Data/Glucose_sim_data_adolescent006_3.npy'
    data_path4 = 'Data/Glucose_sim_data_adolescent006_4.npy'
    data_path5 = 'Data/Glucose_sim_data_adolescent006_5.npy'
    data_path6 = 'Data/Glucose_sim_data_adolescent006_6.npy'
    data_path7 = 'Data/Glucose_sim_data_adolescent006_7.npy'
    data_path8 = 'Data/Glucose_sim_data_adolescent006_8.npy'
    data_path9 = 'Data/Glucose_sim_data_adolescent006_9.npy'



    epochs = 100000
    batch_size = 3000

    hidden_size_alpha = 512
    hidden_size_2 = 128
    hidden_size_4 = 256

    lr = 2e-4
    weight_decay = 1e-3
    load_model_path = 'ckpt/glucose_model_change_isu.pth'

    alpha = 2.  # forward loss weight
    beta = 1.5  # inverse loss weight
    sigma = 2 
