### This script is used to conduct 5-fold validation for hyper-parameter determination for model training.
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
from collections import Counter
from itertools import permutations

### for the tum dataset, do: make 42.5 & 32.5 to be 45 and 35/ for non zero soc range c rate, take 
### average as the charge/ discharge c rate
def generate_data_list_k_fold(efc, k_fold_seed ,train_feat, train_label):


    train_features, train_labels_flaged = train_feat, train_label
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])

    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))

    train_features = np.delete(train_features, feat_to_delete, axis=1)


    '''
    5-fold validation on the public data by randomly split them all
    '''
    np.random.seed(k_fold_seed)
    indices = np.arange(train_features.shape[0])
    np.random.shuffle(indices)  # shuffle

    folds = np.array_split(indices, 5)  # 5 fold

    fold_masks = []
    for i in range(5):
        mask = np.zeros(train_features.shape[0], dtype=int)
        mask[folds[i]] = 1  
        fold_masks.append(mask)


    '''
    added: here we iterate 5 times in order to generate 5 sets of train-validation
    '''
    coordinate_3d_list_train_fold = []
    data_list_train_fold = []
    coordinate_3d_list_val_fold = []
    data_list_val_fold = []    


    for mask in fold_masks:

        train_data_feat = train_features[mask == 0]  
        val_data_feat = train_features[mask == 1]    
        train_data_label = train_labels_flaged[mask == 0]  
        val_data_label = train_labels_flaged[mask == 1]    
    
        coordinate_3d_list_train = []
        data_list_train = []
        coordinate_3d_list_test = []
        data_list_test = []

        test_tem = val_data_feat[:, 28]
        train_tem = train_data_feat[:, 28]
 
        temperature_grid = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])  
        
        ## delete the temperature as a feature
        train_data_feat = np.delete(train_data_feat, 28, axis=1)
        val_data_feat = np.delete(val_data_feat, 28, axis=1)  

        for t in np.unique(temperature_grid):

            if t == 5:
            
                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)

                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                ###################################################################################

                # take all the features and labels of the training and testing data in this temperature
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                #######################################################################################

                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                #############################################################################

                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)

        #################################################################################################################

            if t == 15:

                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)

                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)

                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2

                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################

                # take all the features and labels of the training and testing data in this temperature
                # do not keep the operation features
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]
                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################

                ### now find the unrepeated 2 list of charge and discharge

                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################

                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)


                for j2 in range(unique_test_round.shape[1]):
                    
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 20:

                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]

                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)     

                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)

                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################

                # take all the features and labels of the training and testing data in this temperature
                # do not keep the operation features
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]
                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)

                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################

                ### now find the unrepeated 2 list of charge and discharge

                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################

                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)


                for j2 in range(unique_test_round.shape[1]):
                    
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 25:

                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]

                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)


                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)

                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################

                # take all the features and labels of the training and testing data in this temperature
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]

                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################

                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################

                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)


                for j2 in range(unique_test_round.shape[1]):
                    
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 30:
            
                mask1 = np.isin(train_tem, np.array([t, 35]))  
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)

                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)

                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################

                # take all the features and labels of the training and testing data in this temperature
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]

                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################
                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################
                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)

                for j2 in range(unique_test_round.shape[1]):   
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 35:

                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)
                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)
                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################
                # take all the features and labels of the training and testing data in this temperature
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]

                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################

                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################

                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)

                for j2 in range(unique_test_round.shape[1]):
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 40:
            
                mask1 = np.isin(train_tem, np.array([t, 45]))  
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)
                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)
                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################
                # take all the features and labels of the training and testing data in this temperature
                # do not keep the operation features
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]
                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################
                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################
                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)

                for j2 in range(unique_test_round.shape[1]):
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 45:
                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)

                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)
                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################

                # take all the features and labels of the training and testing data in this temperature
                # do not keep the operation features
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]
                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################

                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################

                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)

                for j2 in range(unique_test_round.shape[1]):
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)

        ##########################################################################################################################

            if t == 50:
                mask1 = np.isin(train_tem, t)
                indices_t1 = np.where(mask1)[0]
                cur_x_train_f = train_data_feat[indices_t1, :]
                cur_y_train_f = train_data_label[indices_t1, :]
                mask2 = np.isin(test_tem, t)
                indices_t2 = np.where(mask2)[0]
                cur_x_test_f = val_data_feat[indices_t2, :]
                cur_y_test_f = val_data_label[indices_t2, :]
                train_charge = cur_x_train_f[:, 28]
                train_discharge = cur_x_train_f[:, 35]
                test_charge = cur_x_test_f[:, 28]
                test_discharge = cur_x_test_f[:, 35]
                train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
                train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
                train_high = np.where(np.abs(train_discharge) > 10)[0]
                test_low1 = np.where(np.abs(test_charge) < 0.001)[0]
                test_low2 = np.where(np.abs(test_discharge) < 0.001)[0]
                train_low = np.union1d(train_low1, train_low2)
                train_low = np.union1d(train_low, train_high)
                test_low = np.union1d(test_low1, test_low2)
                train_charge_f = np.delete(train_charge, train_low)
                train_discharge_f = np.delete(train_discharge, train_low)
                test_charge_f = np.delete(test_charge, test_low)
                test_discharge_f = np.delete(test_discharge, test_low)
                ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
                train_charge_round = np.round(train_charge_f * 2) / 2
                train_discharge_round = np.round(train_discharge_f * 2) / 2
                test_charge_round = np.round(test_charge_f * 2) / 2
                test_discharge_round = np.round(test_discharge_f * 2) / 2
                ###################################################################################
                # take all the features and labels of the training and testing data in this temperature
                # do not keep the operation features
                cur_train_feat = cur_x_train_f[:, 0:28]
                cur_train_lab = cur_y_train_f[:, 0:28]
                cur_test_feat = cur_x_test_f[:, 0:28]
                cur_test_lab = cur_y_test_f[:, 0:28]
                cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
                cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
                cur_test_feat = np.delete(cur_test_feat, test_low, axis=0)
                cur_test_lab = np.delete(cur_test_lab, test_low, axis=0)
                cur_train_data = np.concatenate((cur_train_feat, cur_train_lab), axis=1)
                cur_test_data = np.concatenate((cur_test_feat, cur_test_lab), axis=1)
                #######################################################################################
                ### now find the unrepeated 2 list of charge and discharge
                train_round = np.vstack((train_charge_round, train_discharge_round))
                test_round = np.vstack((test_charge_round, test_discharge_round))
                transposed_array = train_round.T
                unique_rows = np.unique(transposed_array, axis=0)
                unique_train_round = unique_rows.T
                transposed_array2 = test_round.T
                unique_rows2 = np.unique(transposed_array2, axis=0)
                unique_test_round = unique_rows2.T
                #############################################################################
                ### each unique temperature-charge-discharge point contains some data
                for j in range(unique_train_round.shape[1]):
                    cur_charge_c = unique_train_round[0, j]
                    cur_discharge_c = unique_train_round[1, j]
                    data_charge = np.where(train_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(train_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_train_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_train.append(coor_target)
                    data_list_train.append(data_target)
                for j2 in range(unique_test_round.shape[1]):
                    cur_charge_c = unique_test_round[0, j2]
                    cur_discharge_c = unique_test_round[1, j2]
                    data_charge = np.where(test_charge_round == cur_charge_c)[0]
                    data_discharge = np.where(test_discharge_round == cur_discharge_c)[0]
                    data_target_ind = np.intersect1d(data_charge, data_discharge)
                    data_target = cur_test_data[data_target_ind, :]
                    coor_target = np.array([t, cur_charge_c, cur_discharge_c])
                    coordinate_3d_list_test.append(coor_target)
                    data_list_test.append(data_target)
        coordinate_3d_list_train_fold.append(coordinate_3d_list_train)
        coordinate_3d_list_val_fold.append(coordinate_3d_list_test)
        data_list_train_fold.append(data_list_train)
        data_list_val_fold.append(data_list_test)

    return coordinate_3d_list_train_fold, data_list_train_fold, coordinate_3d_list_val_fold, data_list_val_fold



















































