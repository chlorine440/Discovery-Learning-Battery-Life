import torch
import matplotlib.pyplot as plt
import numpy as np

from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.snpe import SNPE_C
from sbi.inference import prepare_for_sbi, simulate_for_sbi

import pickle
from sklearn.model_selection import KFold, cross_validate

from parameter_25R import params_log
from parameter_25R import params_log_flag
from parameter_25R import params_setting

from scipy.interpolate import interp1d

from battery_simulator_p2d import simulator

# Define the device, use GPU if possible
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(torch.cuda.is_available())

# Define some basic parameters for this public from Tongjij
Q_rated = 2.5 # Ah
current = -1.25 # A
dod = 1  # DOD range
V_cut_lb = 2.5
V_cut_ub = 4.2
num_workers = 8

# Define the number of sample trials, 50000 is used in this work
num_simulations=50000
num_dim = 11

t_typ = np.abs(3600*Q_rated*dod/current)
t_sim = np.abs(int(3700*Q_rated*dod/current))

def t_normal(data):
    return data/t_typ
def v_normal(data):
    return (data-V_cut_lb)/(V_cut_ub-V_cut_lb)

prior_min = []
prior_max = []
for i in range(num_dim):
    prior_min.append(list(params_log.values())[i][0])
    prior_max.append(list(params_log.values())[i][1])
prior = utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))


def my_model(params):
    params = np.asarray(params)
    this_param = []
    for j in range(num_dim):
        if list(params_log_flag.values())[j]==1:
            this_param.append(   10**params[j]       )
        else:
            this_param.append(     params[j]         )

    params = params_setting(this_param)
    results = simulator(params, current, t_sim)

    if isinstance(results, str):
        ############# abnormal
        num_points=100
        this_t = np.linspace(0, t_sim, num_points)
        s = np.random.normal(0, 0.005, 100)
        this_v = np.ones(num_points)*V_cut_lb + s
        #############
    else:
        ############# normal
        Voltage = np.array(results["Terminal voltage [V]"].entries)
        Time = np.array(results["Time [s]"].entries)
        # import matplotlib.pyplot as plt
        # plt.plot(Time, Voltage)
        # plt.show()
        num_points=100
        this_t = np.linspace(0, Time[-1], num_points)
        f = interp1d(Time, Voltage, kind='slinear') 
        s = np.random.normal(0, 0.005, 100)
        this_v = f(this_t) + s
        #############

    v_norm = v_normal(this_v)
    t_norm = t_normal(this_t)
    y_obs = np.sqrt(np.square(t_norm)+np.square(v_norm))

    return y_obs

# 1. Implement the data generate
# Data generated are used for the model training
sbi_sim, prior = prepare_for_sbi(my_model, prior)
theta, x = simulate_for_sbi(sbi_sim, proposal=prior, num_simulations=num_simulations, num_workers=num_workers)

### 2.Load the dataset
theta_tra = theta.copy()
x_tra = x.copy()

### 3.Set the hyper-parameters
### max epoch number set to be 200 in this work, to aviod overfitting
### spline flow
flow_step=9
bin=9
tail_bound = 3.0
### resnet
res_block = 9
hidden_feature = int(2**8)
dropout = 0.05
use_batch_norm = True
### training-related
max_num_epochs = 200
batch_size = 256
lr = 0.0005
clip = 21.0
#############################

### 4.Create the model
nde_nsf = posterior_nn(model="nsf" ,z_score_x='structured')

inference = SNPE_C(show_progress_bars=True,prior=prior,density_estimator=nde_nsf, device=device)
inference = inference.append_simulations(theta_tra, x_tra, proposal=prior, data_device=device)

if __name__ == '__main__':

    import time
    start = time.time()
    ### Train the model, validation deactivated
    inference.train(show_train_summary=True, max_num_epochs=max_num_epochs,  
                    learning_rate =lr, training_batch_size=batch_size ,validation_fraction=0, clip_max_norm=clip,
                    )
    end = time.time()
    print("time used:",end-start)
    
    ### Save the trained model
    with open("inference_25R_new_200.pkl", "wb") as handle:
        pickle.dump(inference, handle)


