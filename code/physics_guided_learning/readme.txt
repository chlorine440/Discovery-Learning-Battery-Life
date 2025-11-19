This document gives a brief description about the codes under the 'physics_guided_learning' folder.

(1) First, there is a description about how to modify the original package 'sbi 0.22.0'.
We based parts of our implementation on the publicly available package sbi, version 0.22.0. We made modifications on the following scripts:

'base.py':
Location: sbi\inference\base.py
Changed the get_dataloaders() function, this was originally designed to prepare and output the train_loader and val_loader object. We skip the train-validation split and only output train_loader object. All data are used for training after modification. Instead, early-stopping strategy is used to address overfitting.

'snpe_base.py': 
Location: sbi\inference\snpe\snpe_base.py
Changed the train() function in class PosteriorEstimator, originally the train_loader and val_loader are accessed by the get_loader function, only train_loader is accessed after modification as we described above. We also added a function to record the loss on training data to monitor the training process, since validation are skipped. The calculation of validation data is deleted to reduce computation burden.

The training process for building SBI models can be implemented after substituting the above scripts on the sbi version 0.22.0.

(2) Under each folder, there is a 'readme.txt' documents to illustrate the functionality of different scripts.

(3) The pre-trained simulation-based inference (SBI) models can be accessed on google drive using the link below:
(https://drive.google.com/drive/folders/1AtyKB95ICjrHmxjGX2Nnrhp4WrWcC08r?usp=sharing)

These models can be shared among batteries with similar cell design parameters. These models can be used to sample the posterior distribution of the physical parameters, based on the prior (i.e., uniform distribution) and observation (charge or discharge voltage curve). We name the SBI model documents using the names of commercial models without the names of manufacturers. (Note: the 'PBC' model is shared between PB and PC cells.)
