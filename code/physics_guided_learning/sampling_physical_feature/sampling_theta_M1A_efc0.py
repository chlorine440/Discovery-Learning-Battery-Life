import pickle
import matplotlib.pyplot as plt
from sbi import analysis as analysis
import torch
from sbi import utils as utils
import numpy as np
from scipy.stats import kurtosis, skew

import re
# from parameter_SNL_B import params_log
from parameter_M1A import params_log
import os
import time

def cal_theta(theta_p, theta_n, L_p, L_n, Cs_p, Cs_n, epss_n, epss_p):
    theta = theta_p + theta_n * ( (L_n*Cs_n*epss_n) / (L_p*Cs_p*epss_p))
    return theta

F = 96485.3321 #(Faraday constant)

def cal_delta_theta(q_used, L, A, Cs_max, epss):
    delta_theta = q_used/ (L*A*Cs_max*epss*F)
    return np.abs(delta_theta)


# #############################################################
# from parameter_xxx
Cs_max_n = 30675
Cs_max_p = 22835
area = 794e-4
# Thichness [m]
L_n = 36e-6 
L_p = 81e-6 
current = -0.55 # A

with open("inference_M1A_new_200.pkl", "rb") as handle:
    inference = pickle.load(handle)

directory_volt = "public_efc0_volt/SNL-A" 
directory="public_dataset_arrays_efc0/output_arrays_SNL_A" 

files = os.listdir(directory)

files_volt = os.listdir(directory_volt)

pattern_array = re.compile(r'y_obs_(\d{2})\.npy')

matched_files_array = [f for f in files if pattern_array.match(f)]
matched_files_array.sort(key=lambda x: int(pattern_array.match(x).group(1)))

pattern_volt = re.compile(r'(\d{2})\.npy')

matched_files_volt = [f for f in files_volt if pattern_volt.match(f)]
matched_files_volt.sort(key=lambda x: int(pattern_volt.match(x).group(1)))


###############################################################
num_dim = 11    

# Create a directory to store the new files if it doesn't exist

prior_min = []
prior_max = []
for i in range(num_dim):
    prior_min.append(list(params_log.values())[i][0])
    prior_max.append(list(params_log.values())[i][1])
prior = utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max),device='cuda')
posterior = inference.build_posterior( prior=prior)



feature_array_list = []

for i, file in enumerate(matched_files_array):
    # Check if the file is in .npy format
    # if file.endswith(".npy") and i != 20 and i != 21 and i != 22 and i != 23:
    if file.endswith(".npy"):

        print('dealing with',file)
        cur_volt = matched_files_volt[i]
        print('Current voltage file is:',cur_volt)

        vol = np.load(os.path.join(directory_volt, cur_volt), allow_pickle=True)
        time_start, time_end = vol[:,0][0] , vol[:,0][-1]


        Q_used = current * (time_end-time_start) # in Amper second

        x = np.load(os.path.join(directory, file), allow_pickle=True)
        x = torch.tensor(x)

        posterior_samples = posterior.sample((50000,), x=x.to('cuda'))
        posterior_samples = posterior_samples.to('cpu').detach().numpy()

        cur_theta_p = posterior_samples[:,10]
        cur_theta_n = posterior_samples[:,9]
        cur_eps_p = posterior_samples[:,8]
        cur_eps_n = posterior_samples[:,7]

        delta_theta_n = cal_delta_theta(Q_used, L_n, area, Cs_max_n, cur_eps_n)
        delta_theta_p = cal_delta_theta(Q_used, L_p, area, Cs_max_p, cur_eps_p)

        if current > 0: #discharging

            theta_low_n = cur_theta_n - delta_theta_n
            theta_high_n = cur_theta_n
            theta_low_p = cur_theta_p + delta_theta_p
            theta_high_p = cur_theta_p

        if current < 0: #charging

            theta_low_n = cur_theta_n
            theta_high_n = cur_theta_n + delta_theta_n
            theta_low_p = cur_theta_p
            theta_high_p = cur_theta_p - delta_theta_p

        theta_high_n,theta_low_n,theta_high_p,theta_low_p=theta_high_n.reshape(-1, 1),theta_low_n.reshape(-1, 1),theta_high_p.reshape(-1, 1),theta_low_p.reshape(-1, 1)

        result_theta = cal_theta(cur_theta_p, cur_theta_n, L_p, L_n, Cs_max_p, Cs_max_n, cur_eps_n, cur_eps_p)
        result_theta = result_theta.reshape(-1, 1)

        feature_array = posterior_samples[:, :-2]
        # feature_array = posterior_samples

        feature_array = np.hstack((feature_array, theta_high_n, theta_low_n,theta_high_p, theta_low_p, result_theta))

        _ = analysis.pairplot(feature_array, figsize=(5, 5))# plot posterior samples
        plt.show()

        '''
        Can save this individual cell's feature sampl  e result if needed
        '''
        np.save(f'SNL_A_0_cell_{i}_sample_result')        

        # Calculate statistics for each column
        mean_values = np.mean(feature_array, axis=0, dtype=np.float64)  # Mean along columns
        std_values = np.std(feature_array, axis=0, ddof=1, dtype=np.float64)  # Sample standard deviation along columns
        kurtosis_values = kurtosis(feature_array, axis=0, fisher=False)  # Kurtosis along columns
        skewness_values = skew(feature_array, axis=0)  # Skewness along columns

        # Concatenate all the statistics into a single array
        statistics = np.concatenate((mean_values, std_values, kurtosis_values, skewness_values), axis=None)
        feature_array_list.append(statistics)

        print('dealing with',file)

cur_dataset_feature = np.vstack(feature_array_list)
# np.save('sbi_feature_SNL_A_efc0.npy', cur_dataset_feature)

print(cur_dataset_feature)
print(cur_dataset_feature.shape)
      