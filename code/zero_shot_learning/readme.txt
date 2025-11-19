This document gives a brief description about the codes under the 'zero_shot_learning' folder. 
(1) The 'data_preparation.py' script is used to process the features: extracting the 'mean at efc0' and 'variation in mean value between efc 0 and efc 50'. The 'efc' here denotes equivalent full cycle.
(2) The 'data_preparation_k_fold.py' script is used to conduct 5-fold validation for hyper-parameter determination for model training. 
(3) The 'main.py' script is used for the training and testing of the dual-predictor structure, involving both the base predictor and the meta predictor.
It should be noted that optuna library is used 'main.py'; if the recording function is tured on, the 'db_k_fold.sqlite3' file needs to be deleted before the next run.
(4) Under each folder, there is a 'readme.txt' document to illustrate the functionality of different scripts.


