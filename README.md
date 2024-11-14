run on python 3.9.7

conda create -n BNN python=3.9.7
conda activate BNN
pip install -r requirements.txt

```
├── ckpt/               # pre-trained model
├── Data/
│   ├── generate_Glucose_sim.py      # simulator 
│   ├── vpatient_params.csv
│   └── Glucose_sim_data006_(0-9).npy    # OOD data
├── baseline/
├── casual_tree/
│   ├── Tree.py
│   └── Graph.py
|── RL # RL experiments
├── Net
│   ├── couple_layers.py     # Real_NVP
│   ├── OWN.py               # OWN
│   └── OWN_Linear           # OWN
├── G2M_model.py        # BNN and forward
└── main.py
 ```

time series prediction example:
CUDA_VISIBLE_DEVICES=0 nohup python main.py --epochs 10000 --lr 5e-4 --log_dir ./log/lr_5e-4 >> 5e-4.log &

RL example
cd RL
CUDA_VISIBLE_DEVICES=0 nohup python run.py --experiment_name 006_20_0_450_lr_1e-4_f_32_h_32f_1e-2_4  --patient_id 006 --carbon 20 --reset_low 0 --reset_high 450 --eta 1e-2 --fn 32 32 --bn 32 32 --hidden 32 0  --load_path 006lr_1e-4_f_32_h_32f.pth