Access to public data may require the permission from the original author. In such case, if you need the raw data and processed data for your study, please contact the related corresponding author first.
(1) 'parameter_XXX.py' is used to provide information about the parameter settings and boundaries.
(2) 'public_dataset_arrays' includes the normalized observation vectors that will be the input to the sbi models. (efc0 or efc50 are for equivalent-full cycle 0 or 50)
(3) 'public_efc0_volt' or 'public_efc50_volt' includes the unnormalized time-voltage information.
(4) 'sampling_theta_XXX_efc0' or 'sampling_theta_XXX_efc50' is used to extract the physical features. The 'theta' here denotes the physical parameter vectors, instead of the electrode stoichiometry. This process generally includes two steps: sampling the posterior distribution of the first 11 physical parameters, and then calculating the distribution of the last 3 parameters.

We note that due to commercial confidentiality, certain commercially sensitive cell design information are temporarily not disclosed (the design parameters of PA, PB, PC, and PD cells). 

For public battery cells, the physical feature results from the Interpreter can be obtained by using the saved sbi models. (see the 'sampling_physical_feature' folder)

For public battery cells and our large-format cells, the extracted raw physical features can be directly used for reproducing the paper's results. (see the 'zero_shot_learning' folder --> the 'raw_feature' folder)

