### This script is used to process the features: extracting the 'mean at efc0' and 'variation in mean value between efc 0 and efc 50'. 'efc' here denotes equivalent full cycle.

import numpy as np

#######################################################################################
###You can choose which batteries to predict, by changing the data to be loaded.
test_data_efc0 = np.load('/data/raw_feature/test_data_efc0_26groups.npy')
test_data_efc50 = np.load('/data/raw_feature/test_data_efc50_26groups.npy')

data_tongji_efc0 = np.load('/data/raw_feature/data_25r_efc0.npy')
data_tongji_efc50 = np.load('/data/raw_feature/data_25r_efc50.npy')

data_snl_a_efc0 = np.load('/data/raw_feature/data_m1a_efc0.npy')
data_snl_a_efc50 = np.load('/data/raw_feature/data_m1a_efc50.npy')

data_snl_b_efc0 = np.load('/data/raw_feature/data_hg2_efc0.npy')
data_snl_b_efc50 = np.load('/data/raw_feature/data_hg2_efc50.npy')

data_elt_efc0 = np.load('/data/raw_feature/data_mj1_efc0.npy')
data_elt_efc50 = np.load('/data/raw_feature/data_mj1_efc50.npy')

data_tum_efc0 = np.load('/data/raw_feature/data_vtc5a_efc0.npy')
data_tum_efc50 = np.load('/data/raw_feature/data_vtc5a_efc50.npy')

data_cmu_efc0 = np.load('/data/raw_feature/data_vtc6_efc0.npy')
data_cmu_efc50 = np.load('/data/raw_feature/data_vtc6_efc50.npy')

########################################################################################
### for the tum dataset, do: make 42.5 & 32.5 to be 45 and 35/ for non zero soc range c rate, take 
### average as the charge/ discharge c rate
def generate_data_list_group_mape(efc):
    
    ### save the 3-D operational conditions coordinates and the corresponding data cluster to the list
    ### the data in the lists are used for the predictor
    coordinate_3d_list_train = []
    data_list_train = []
    coordinate_3d_list_test = []
    data_list_test = []

    label_tum = data_tum_efc0[:, 74:].copy()
    label_cmu = data_cmu_efc0[:, 74:].copy()
    label_snl_a = data_snl_a_efc0[:, 74:].copy()
    label_snl_b = data_snl_b_efc0[:, 74:].copy()
    label_tongji = data_tongji_efc0[:, 74:].copy()
    label_elt = data_elt_efc0[:, 74:].copy()
    label_test = test_data_efc0[:, 74:].copy()

    if efc == 0:

        data_tum = data_tum_efc0[:, :74].copy()
        data_cmu = data_cmu_efc0[:, :74].copy()
        data_snl_a = data_snl_a_efc0[:, :74].copy()
        data_snl_b = data_snl_b_efc0[:, :74].copy()
        data_tongji = data_tongji_efc0[:, :74].copy()
        data_elt = data_elt_efc0[:, :74].copy()
        data_test = test_data_efc0[:, :74].copy()

    if efc == 50:

        data_tum = data_tum_efc50.copy()
        data_cmu = data_cmu_efc50.copy()
        data_snl_a = data_snl_a_efc50.copy()
        data_snl_b = data_snl_b_efc50.copy()
        data_tongji = data_tongji_efc50.copy()
        data_elt = data_elt_efc50.copy()
        data_test = test_data_efc50.copy()

    ### prepare the data for the base-predictor and meta-predictor
    ### round the temperature to the nearest multiply of 5 degrees, round the average C-rate to the nearest 0.5C
    mean_in_charge = np.array([58,60,62,64])
    mean_in_discharge = np.array([66,68,70,72])




























    source_features_mod = data_tum.copy()
    source_labels = label_tum.copy()

    train_tem = source_features_mod[:, 56]
    source_features_mod[:, 56] = 5 * np.round((train_tem + 1e-9) / 5)

    cur_feat = source_features_mod.copy()
    char_mean = np.where(cur_feat[:, mean_in_charge] != 0, cur_feat[:, mean_in_charge], np.nan)
    dischar_mean = np.where(cur_feat[:, mean_in_discharge] != 0, cur_feat[:, mean_in_discharge], np.nan)

    char_c = 0.5 * np.round((np.nanmean(char_mean, axis=1) + 1e-9) / 0.5)
    dischar_c = 0.5 * np.round((np.nanmean(dischar_mean, axis=1) + 1e-9) / 0.5)
    char_c[char_c == 0] = 0.5
    dischar_c[dischar_c == 0] = 0.5

    source_features_mod[:, 58] = char_c
    source_features_mod[:, 72] = dischar_c

    train_features = source_features_mod.copy()
    train_labels = source_labels.copy()
    train_flag1 = np.full(train_features.shape[0], 1, dtype=int)
    train_labels_flaged = np.column_stack((train_labels, train_flag1))

    ##### Delete the cell with life less than 70, for the training cells
    indices = np.where(train_labels < 70 )[0]
    train_features = np.delete(train_features, indices, axis=0)
    train_labels_flaged = np.delete(train_labels_flaged, indices, axis=0)
    ### record the data
    if efc == 0:
        train_feat_tum_0 = train_features.copy()
        train_label_tum_0 = train_labels_flaged.copy()
    if efc == 50:
        train_feat_tum_50 = train_features.copy()
        train_label_tum_50 = train_labels_flaged.copy()
    ###############################################################################
    ### delete the redundant features, e.g. skew, kur, tem_std, operation_std
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])
    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))
    train_features = np.delete(train_features, feat_to_delete, axis=1)
    train_tem = train_features[:, 28]

    ## delete the temperature as a feature
    train_features = np.delete(train_features, 28, axis=1)
    
    ### for the low temperature (-5), we put it in the 5 degree point since we dont have below 0 in public dataset
    train_tem[train_tem == -5] = 5
    train_tem[train_tem == 13.75] = 15
    train_tem[train_tem == 32.5] = 35

    target_temperature_mesh = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])

    ### for each of the temperature meshes, cluster the cells by C-rate
    for t in np.unique(target_temperature_mesh):

        if t == 5:
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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



        if t == 15:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)

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

    ##########################################################################################################################

        if t == 20:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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


    ##########################################################################################################################

        if t == 25:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]


            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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

    ##########################################################################################################################

        if t == 30:

            mask1 = np.isin(train_tem, np.array([t, 35]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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
    ##########################################################################################################################

        if t == 35:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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

    ##########################################################################################################################

        if t == 40:
        
            mask1 = np.isin(train_tem, np.array([t, 45]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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

    ##########################################################################################################################

        if t == 45:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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

    ##########################################################################################################################

        if t == 50:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]

            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
            # cur_train_feat = np.delete(cur_train_feat, train_low, axis=0)
            # cur_train_lab = np.delete(cur_train_lab, train_low, axis=0)
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


#######################################################################################################

    source_features_mod = data_cmu.copy()
    source_labels = label_cmu.copy()

    train_tem = source_features_mod[:, 56]
    source_features_mod[:, 56] = 5 * np.round((train_tem + 1e-9) / 5)

    train_features = source_features_mod.copy()
    train_labels = source_labels.copy()
    train_flag1 = np.full(train_features.shape[0], 1, dtype=int)
    train_labels_flaged = np.column_stack((train_labels, train_flag1))

    ##### Delete the cell with life less than 70, for the training cells
    indices = np.where(train_labels < 70 )[0]
    train_features = np.delete(train_features, indices, axis=0)
    train_labels_flaged = np.delete(train_labels_flaged, indices, axis=0)
    ### record the data
    if efc == 0:
        train_feat_cmu_0 = train_features.copy()
        train_label_cmu_0 = train_labels_flaged.copy()
    if efc == 50:
        train_feat_cmu_50 = train_features.copy()
        train_label_cmu_50 = train_labels_flaged.copy()
    ###############################################################################
    ### delete the redundant features, e.g. skew, kur, tem_std, operation_std
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])
    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))
    train_features = np.delete(train_features, feat_to_delete, axis=1)
    train_tem = train_features[:, 28]

    ## delete the temperature as a feature
    train_features = np.delete(train_features, 28, axis=1)
    
    ### for the low temperature (-5), we put it in the 5 degree point since we dont have below 0 in public dataset
    train_tem[train_tem == -5] = 5
    train_tem[train_tem == 13.75] = 15
    train_tem[train_tem == 32.5] = 35

    target_temperature_mesh = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])

    ### for each of the temperature meshes, cluster the cells by C-rate
    for t in np.unique(target_temperature_mesh):

        if t == 5:
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
            train_charge_f = np.delete(train_charge, train_low)
            train_discharge_f = np.delete(train_discharge, train_low)
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge_f * 2) / 2
            train_discharge_round = np.round(train_discharge_f * 2) / 2
            ###################################################################################
            # take all the features and labels of the data in this temperature
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

        if t == 15:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 20:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################       
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


    ##########################################################################################################################

        if t == 25:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 30:

            mask1 = np.isin(train_tem, np.array([t, 35]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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
    ##########################################################################################################################

        if t == 35:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 40:
        
            mask1 = np.isin(train_tem, np.array([t, 45]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 45:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 50:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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


    data_snl = np.vstack((data_snl_a, data_snl_b))
    label_snl = np.vstack((label_snl_a, label_snl_b))
    source_features_mod = data_snl.copy()
    source_labels = label_snl.copy()

    train_tem = source_features_mod[:, 56]
    source_features_mod[:, 56] = 5 * np.round((train_tem + 1e-9) / 5)

    train_features = source_features_mod.copy()
    train_labels = source_labels.copy()
    train_flag1 = np.full(train_features.shape[0], 1, dtype=int)
    train_labels_flaged = np.column_stack((train_labels, train_flag1))

    ##### Delete the cell with life less than 70, for the training cells
    indices = np.where(train_labels < 70 )[0]
    train_features = np.delete(train_features, indices, axis=0)
    train_labels_flaged = np.delete(train_labels_flaged, indices, axis=0)
    ### record the data
    if efc == 0:
        train_feat_snl_0 = train_features.copy()
        train_label_snl_0 = train_labels_flaged.copy()
    if efc == 50:
        train_feat_snl_50 = train_features.copy()
        train_label_snl_50 = train_labels_flaged.copy()
    ###############################################################################
    ### delete the redundant features, e.g. skew, kur, tem_std, operation_std
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])
    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))
    train_features = np.delete(train_features, feat_to_delete, axis=1)
    train_tem = train_features[:, 28]

    ## delete the temperature as a feature
    train_features = np.delete(train_features, 28, axis=1)
    
    ### for the low temperature (-5), we put it in the 5 degree point since we dont have below 0 in public dataset
    train_tem[train_tem == -5] = 5
    train_tem[train_tem == 13.75] = 15
    train_tem[train_tem == 32.5] = 35

    target_temperature_mesh = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])

    ### for each of the temperature meshes, cluster the cells by C-rate
    for t in np.unique(target_temperature_mesh):

        if t == 5:
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
            train_charge_f = np.delete(train_charge, train_low)
            train_discharge_f = np.delete(train_discharge, train_low)
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge_f * 2) / 2
            train_discharge_round = np.round(train_discharge_f * 2) / 2
            ###################################################################################
            # take all the features and labels of the data in this temperature
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

        if t == 15:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 20:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################       
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


    ##########################################################################################################################

        if t == 25:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 30:

            mask1 = np.isin(train_tem, np.array([t, 35]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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
    ##########################################################################################################################

        if t == 35:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 40:
        
            mask1 = np.isin(train_tem, np.array([t, 45]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 45:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 50:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    source_features_mod = data_elt.copy()
    source_labels = label_elt.copy()

    train_tem = source_features_mod[:, 56]
    source_features_mod[:, 56] = 5 * np.round((train_tem + 1e-9) / 5)

    train_features = source_features_mod.copy()
    train_labels = source_labels.copy()
    train_flag1 = np.full(train_features.shape[0], 1, dtype=int)
    train_labels_flaged = np.column_stack((train_labels, train_flag1))

    ##### Delete the cell with life less than 70, for the training cells
    indices = np.where(train_labels < 70 )[0]
    train_features = np.delete(train_features, indices, axis=0)
    train_labels_flaged = np.delete(train_labels_flaged, indices, axis=0)
    ### record the data
    if efc == 0:
        train_feat_elt_0 = train_features.copy()
        train_label_elt_0 = train_labels_flaged.copy()
    if efc == 50:
        train_feat_elt_50 = train_features.copy()
        train_label_elt_50 = train_labels_flaged.copy()
    ###############################################################################
    ### delete the redundant features, e.g. skew, kur, tem_std, operation_std
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])
    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))
    train_features = np.delete(train_features, feat_to_delete, axis=1)
    train_tem = train_features[:, 28]

    ## delete the temperature as a feature
    train_features = np.delete(train_features, 28, axis=1)
    
    ### for the low temperature (-5), we put it in the 5 degree point since we dont have below 0 in public dataset
    train_tem[train_tem == -5] = 5
    train_tem[train_tem == 13.75] = 15
    train_tem[train_tem == 32.5] = 35

    target_temperature_mesh = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])

    ### for each of the temperature meshes, cluster the cells by C-rate
    for t in np.unique(target_temperature_mesh):

        if t == 5:
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
            train_charge_f = np.delete(train_charge, train_low)
            train_discharge_f = np.delete(train_discharge, train_low)
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge_f * 2) / 2
            train_discharge_round = np.round(train_discharge_f * 2) / 2
            ###################################################################################
            # take all the features and labels of the data in this temperature
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

        if t == 15:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 20:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################       
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


    ##########################################################################################################################

        if t == 25:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 30:

            mask1 = np.isin(train_tem, np.array([t, 35]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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
    ##########################################################################################################################

        if t == 35:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 40:
        
            mask1 = np.isin(train_tem, np.array([t, 45]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 45:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 50:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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


    source_features_mod = data_tongji.copy()
    source_labels = label_tongji.copy()

    train_tem = source_features_mod[:, 56]
    source_features_mod[:, 56] = 5 * np.round((train_tem + 1e-9) / 5)

    train_features = source_features_mod.copy()
    train_labels = source_labels.copy()
    train_flag1 = np.full(train_features.shape[0], 1, dtype=int)
    train_labels_flaged = np.column_stack((train_labels, train_flag1))

    ##### Delete the cell with life less than 70, for the training cells
    indices = np.where(train_labels < 70 )[0]
    train_features = np.delete(train_features, indices, axis=0)
    train_labels_flaged = np.delete(train_labels_flaged, indices, axis=0)
    ### record the data
    if efc == 0:
        train_feat_tongji_0 = train_features.copy()
        train_label_tongji_0 = train_labels_flaged.copy()
    if efc == 50:
        train_feat_tongji_50 = train_features.copy()
        train_label_tongji_50 = train_labels_flaged.copy()
    ###############################################################################
    ### delete the redundant features, e.g. skew, kur, tem_std, operation_std
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])
    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))
    train_features = np.delete(train_features, feat_to_delete, axis=1)
    train_tem = train_features[:, 28]

    ## delete the temperature as a feature
    train_features = np.delete(train_features, 28, axis=1)
    
    ### for the low temperature (-5), we put it in the 5 degree point since we dont have below 0 in public dataset
    train_tem[train_tem == -5] = 5
    train_tem[train_tem == 13.75] = 15
    train_tem[train_tem == 32.5] = 35

    target_temperature_mesh = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])

    ### for each of the temperature meshes, cluster the cells by C-rate
    for t in np.unique(target_temperature_mesh):

        if t == 5:
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
            train_charge_f = np.delete(train_charge, train_low)
            train_discharge_f = np.delete(train_discharge, train_low)
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge_f * 2) / 2
            train_discharge_round = np.round(train_discharge_f * 2) / 2
            ###################################################################################
            # take all the features and labels of the data in this temperature
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

        if t == 15:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 20:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)
            #####################################################
            ####################################################       
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


    ##########################################################################################################################

        if t == 25:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 30:

            mask1 = np.isin(train_tem, np.array([t, 35]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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
    ##########################################################################################################################

        if t == 35:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 40:
        
            mask1 = np.isin(train_tem, np.array([t, 45]))  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 45:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    ##########################################################################################################################

        if t == 50:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            train_low1 = np.where(np.abs(train_charge) < 0.001)[0]
            train_low2 = np.where(np.abs(train_discharge) < 0.001)[0]
            train_high = np.where(np.abs(train_discharge) > 10)[0]
            train_low = np.union1d(train_low1, train_low2)
            train_low = np.union1d(train_low, train_high)

            #####################################################
            ####################################################
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

    source_features_mod = data_test.copy()
    source_labels = label_test.copy()

    train_tem = source_features_mod[:, 56]
    source_features_mod[:, 56] = 5 * np.round((train_tem + 1e-9) / 5)

    train_features = source_features_mod.copy()
    train_labels = source_labels.copy()
    train_flag1 = np.full(train_features.shape[0], 1, dtype=int)
    train_labels_flaged = np.column_stack((train_labels, train_flag1))

    ### delete the redundant features, e.g. skew, kur, tem_std, operation_std
    sbi_kur_gate = np.arange(28,42)
    sbi_skew_gate = np.arange(42,56)
    tem_std_gate = np.array([57])
    operation_std_gate = np.array([59,61,63,65, 67,69,71,73])
    feat_to_delete = np.concatenate((sbi_kur_gate, sbi_skew_gate, tem_std_gate, operation_std_gate))
    train_features = np.delete(train_features, feat_to_delete, axis=1)
    train_tem = train_features[:, 28]

    ## delete the temperature as a feature
    train_features = np.delete(train_features, 28, axis=1)
    
    ### for the low temperature (-5), we put it in the 5 degree point since we dont have below 0 in public dataset
    train_tem[train_tem == -5] = 5
    train_tem[train_tem == 13.75] = 15
    train_tem[train_tem == 32.5] = 35

    target_temperature_mesh = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50])

    ### for each of the temperature meshes, cluster the cells by C-rate
    for t in np.unique(target_temperature_mesh):

        if t == 5:
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

        if t == 15:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

    ##########################################################################################################################

        if t == 20:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            #####################################################
            ####################################################       
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)


    ##########################################################################################################################

        if t == 25:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

    ##########################################################################################################################

        if t == 30:

            mask1 = np.isin(train_tem, t)  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)
    ##########################################################################################################################

        if t == 35:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################

            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

    ##########################################################################################################################

        if t == 40:
        
            mask1 = np.isin(train_tem, t)  
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

    ##########################################################################################################################

        if t == 45:
        
            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]

            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

    ##########################################################################################################################

        if t == 50:

            mask1 = np.isin(train_tem, t)
            indices_t1 = np.where(mask1)[0]
            cur_x_train_f = train_features[indices_t1, :]
            cur_y_train_f = train_labels_flaged[indices_t1, :]
            train_charge = cur_x_train_f[:, 28]
            train_discharge = cur_x_train_f[:, 35]
            #####################################################
            ####################################################
            ############### here, make the charge and discharge current to be the closest 0.5 multiple ###
            train_charge_round = np.round(train_charge * 2) / 2
            train_discharge_round = np.round(train_discharge * 2) / 2
            ###################################################################################
            # take all the features and labels of the training and testing data in this temperature
            cur_train_feat = cur_x_train_f[:, 0:28]
            cur_train_lab = cur_y_train_f[:, 0:28]
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
                coordinate_3d_list_test.append(coor_target)
                data_list_test.append(data_target)

    if efc == 0:

        train_feat = np.vstack((train_feat_tongji_0,
                                train_feat_snl_0,
                                train_feat_elt_0, 
                                train_feat_tum_0,
                                train_feat_cmu_0))

        train_label = np.vstack((train_label_tongji_0,
                                train_label_snl_0,
                                train_label_elt_0, 
                                train_label_tum_0,
                                train_label_cmu_0))

    if efc == 50:

        train_feat = np.vstack((train_feat_tongji_50,
                                train_feat_snl_50,
                                train_feat_elt_50, 
                                train_feat_tum_50,
                                train_feat_cmu_50))

        train_label = np.vstack((train_label_tongji_50,
                                train_label_snl_50,
                                train_label_elt_50, 
                                train_label_tum_50,
                                train_label_cmu_50))

    combined = list(zip(coordinate_3d_list_train, data_list_train))
    sorted_combined = sorted(combined, key=lambda x: x[0][0]) 
    sorted_list1, sorted_list2 = zip(*sorted_combined)
    coordinate_3d_list_train = list(sorted_list1)
    data_list_train = list(sorted_list2)

    combinedtest = list(zip(coordinate_3d_list_test, data_list_test))
    sorted_combinedtest = sorted(combinedtest, key=lambda x: x[0][0]) 
    sorted_list1test, sorted_list2test = zip(*sorted_combinedtest)
    coordinate_3d_list_test = list(sorted_list1test)
    data_list_test = list(sorted_list2test)

    return coordinate_3d_list_train, data_list_train, coordinate_3d_list_test, data_list_test, train_feat, train_label



















































