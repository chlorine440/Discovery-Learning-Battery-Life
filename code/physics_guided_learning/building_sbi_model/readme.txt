(1) 'battery_simulator_p2d.py' uses the 'PyBaMM' package to build the pseudo-two-dimensional (P2D) model.
(2) 'parameter_XXX.py' sets the cell design and material property parameters either from measurements or previous literature.
(3) 'generate_model_XXX.py' uses the 'sbi' package to build the sbi models. Basically, it includes three steps: sampling physical parameters, training sbi model, saving sbi model. We keep the hyperparameters consistent over different battery types.

We note that due to commercial confidentiality, certain commercially sensitive cell design information are temporarily not disclosed (the design parameters of PA, PB, PC, and PD cells). They may be avaiable from corresponding authors upon request and subject to agreement. 

For public battery cells, the physical feature results from the Interpreter can be obtained by using the saved sbi models. (see the 'sampling_physical_feature' folder)

For public battery cells and our large-format cells, the extracted raw physical features can be directly used for reproducing the paper's results. (see the 'zero_shot_learning' folder --> the 'raw_feature' folder)
