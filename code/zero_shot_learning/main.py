# this code generate the predictor results, for the meta-predictor and the base-predictor
# data processing, model selection, hyperparameters, are described in supplementary materials

# please use same package versions with this work (optuna, numpy, pandas, sk-learn, SBI)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from data_preparation_k_fold import generate_data_list_k_fold
from data_preparation import generate_data_list_group_mape

from collections import defaultdict
import pandas as pd

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)


def Testing():

    ### In this function we search optimal hyperparameters on 5-fold validation, then run the prediction
    study_name = f'search_record_for_k_fold'
    storage_url = f'sqlite:///db_k_fold.sqlite3'
    
    # define the searching space for the hyperparameters
    search_space = {
        "alpha_f_sbi": np.arange(0.8, 1.2, 0.1).tolist(), 
        "C_f_ope": np.arange(0.9, 1.5, 0.1).tolist(),   
        "epsilon_f_ope": np.arange(0.1, 0.3, 0.1).tolist(), 
        "l1_ratio": np.arange(0.4, 0.9, 0.1).tolist(),   
        } # 4 x 6 x 2 x 4 = 192
    
    # we use the grid sampler, search to the end, since the target is the global optimal under current setting
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    # create the optuna searching task, can load existing study, if already performed the study once
    study = optuna.create_study(study_name=study_name,direction="maximize",sampler=sampler) 
    c1, d1, c2, d2, train_feat_0, train_label_0 = generate_data_list_group_mape(0)
    c1, d1, c2, d2, train_feat_50, train_label_50 = generate_data_list_group_mape(50)
    objective_with_params = lambda trial: objective_linear_k_fold(trial,train_feat_0, train_label_0, train_feat_50, train_label_50)
    study.optimize(objective_with_params, n_trials=192)

    # save the result optimal hyperparameters under 5-fold validation
    trial = study.best_trial
    alpha_f_sbi, C_f_ope, epsilon_f_ope, l1_ratio = trial.params['alpha_f_sbi'],trial.params['C_f_ope'],trial.params['epsilon_f_ope'], trial.params['l1_ratio']
    print(alpha_f_sbi, C_f_ope, epsilon_f_ope, l1_ratio)

    # alpha_f_sbi, C_f_ope, epsilon_f_ope, l1_ratio =0.8, 1.4 ,0.2, 0.8
    # use the obtained optimal values on the prediction process of testing data
    cur_test_mape, cur_group_mape = objective_linear_test(alpha_f_sbi, C_f_ope, epsilon_f_ope, l1_ratio)
    print(f'Group mape = {cur_group_mape} finished!')
    return 


def train_test_linear_model(coordinate_3d_list_train, data_list_train, coordinate_3d_list_test, data_list_test,
                            alpha_f_sbi, C_f_ope, epsilon_f_ope, feat_num,  l1_ratio):
    
    ### train and save a model (sklearn-elastic net) for each existing training data cluster
    model_parameters_train = []
    for i in range (len(coordinate_3d_list_train)):

        train_data = data_list_train[i]
        train_feature = train_data[:, :feat_num]
        train_efc = train_data[:, 28]

        ############################################################################
        # set the bad training features to be zeros or not
        k_value = 10 if train_feature.shape[1] > 10 else 5 if train_feature.shape[1] > 6 else 3 if train_feature.shape[1] > 3 else 1
        selector_f_regression = SelectKBest(score_func=mutual_info_regression, k=k_value) 
        selector_f_regression.fit(train_feature, train_efc)
        feat_ex = np.argsort(-selector_f_regression.scores_)[(train_feature.shape[0]) - 1 :]
        train_feature[:, feat_ex] = 0
        ############################################################################

        # Train the Elastic-Net model for each operational condition cluster, record the model parameters
        f_sbi = linear_model.ElasticNet(alpha = alpha_f_sbi, random_state=0, l1_ratio=l1_ratio)
        f_sbi.fit(train_feature, train_efc)
        coefficients = f_sbi.coef_
        intercept = f_sbi.intercept_  
        parameter = np.insert(coefficients, 0, intercept)
        model_parameters_train.append(parameter)

    f_ope_output = np.array(model_parameters_train)
    f_ope_input = np.array(coordinate_3d_list_train)

    ##### in this work we use a SVR model for the f-ope
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    test_coordinate_input = np.array(coordinate_3d_list_test)

    X_train = scaler_X.fit_transform(f_ope_input)
    X_test = scaler_X.transform(test_coordinate_input)
    y_train = scaler_y.fit_transform(f_ope_output)

    svr = SVR(kernel = 'rbf', C = C_f_ope, epsilon = epsilon_f_ope)
    f_ope = MultiOutputRegressor(svr)
    f_ope.fit(X_train, y_train)

    # Base-predictor's parameter obtained by SVR model's prediction
    y_pred = f_ope.predict(X_test)
    model_parameters_test = scaler_y.inverse_transform(y_pred)

    # Record the error of testing data, for individual cells or operational condition groups.
    test_efc_list = []
    test_pred_list = []

    group_sums_efc = defaultdict(float)
    group_sums_pred = defaultdict(float)
    group_counts = defaultdict(int)    

    group_sum_sbi = defaultdict(lambda: np.zeros(28)) 
    
    record_efc = []
    record_group = []
    record_pred = []
    record_ind = []

    for j in range (len(coordinate_3d_list_test)):

        cur_parameter = model_parameters_test[j, :]
        test_gpr = linear_model.ElasticNet(alpha=1, random_state=0)
        test_gpr.coef_ = cur_parameter[1:]
        test_gpr.intercept_ = cur_parameter[0]

        cur_test_data = data_list_test[j]
        cur_test_sbi = cur_test_data[:, :feat_num]
        cur_test_efc = cur_test_data[:, 28]
        cur_test_group = cur_test_data[:, 29]
        cur_test_ind = cur_test_data[:, 30]

        cur_test_pred = np.abs(test_gpr.predict(cur_test_sbi))

        record_efc.append(cur_test_efc)
        record_group.append(cur_test_group)
        record_pred.append(cur_test_pred)
        record_ind.append(cur_test_ind)

        # record the error for each group
        for k in range(len(cur_test_efc)):
            index = cur_test_group[k]
            value_efc = cur_test_efc[k]
            value_pred = cur_test_pred[k]
            value_sbi = cur_test_sbi[k, :]

            group_sums_efc[index] += value_efc
            group_sums_pred[index] += value_pred
            group_counts[index] += 1
            group_sum_sbi[index] += value_sbi

        for points_num in range(len(cur_test_pred)):
            test_efc_list.append(cur_test_efc[points_num])
            test_pred_list.append(cur_test_pred[points_num])

    ### Output the prediction result of testing cells to be an excel file if needed.
    record_efc = np.concatenate(record_efc)
    record_pred = np.concatenate(record_pred)

    df = pd.DataFrame({
        'true efc': record_efc,
        'predicted efc': record_pred
    })

    # # Save to an Excel file
    # df.to_excel("output_.xlsx", index=False)    

    # Save & visualize the prediction results if needed.
    average_values = []
    average_sbi = []
    for index in sorted(group_sums_efc.keys()):
        average_value_efc = group_sums_efc[index] / group_counts[index]
        average_value_pred = group_sums_pred[index] / group_counts[index]
        average_values.append([average_value_efc, average_value_pred,  index])     

        ave_value_sbi = group_sum_sbi[index] / group_counts[index]
        average_sbi.append(ave_value_sbi) 

    average_values_array = np.array(average_values)
    group_loss_list = np.abs(average_values_array[:, 0] - average_values_array[:, 1]) / average_values_array[:, 0]
    group_mape = np.mean(group_loss_list)    
    test_efc_list = np.array(test_efc_list)
    test_pred_list = np.array(test_pred_list)
    loss_list = np.abs(test_efc_list - test_pred_list) / test_efc_list
    mape = np.mean(loss_list)

    plt.subplot(1, 2, 1)
    x2= np.arange(len(loss_list))+1
    plt.plot(x2, loss_list)
    plt.scatter(x2, loss_list ,s=5,c='r')

    plt.subplot(1, 2, 2)
    ref = np.arange(0,max(max(test_pred_list),max(test_efc_list)),0.1)
    plt.plot(ref,ref,c='k')
    plt.scatter(test_pred_list, test_efc_list, c ='r')

    np.save('cell_predicted_value_63.npy', test_pred_list)
    np.save('cell_observed_value_63.npy', test_efc_list)

    plt.ylabel('Observed values',fontsize=16)
    plt.xlabel('Predicted values',fontsize=16)
    plt.title(f'cell loss, MAPE={round(mape,4)}')
    plt.show()     


    plt.subplot(1, 2, 1)
    x2= np.arange(len(group_loss_list))+1
    plt.plot(x2, group_loss_list)
    plt.scatter(x2, group_loss_list ,s=5,c='r')

    plt.subplot(1, 2, 2)
    ref = np.arange(0,max(max(average_values_array[:, 1]),max(average_values_array[:, 0])),0.1)
    plt.plot(ref,ref,c='k')
    plt.scatter(average_values_array[:, 1], average_values_array[:, 0], c ='r')

    np.save('group_predicted_value_26.npy', average_values_array[:, 1])
    np.save('group_observed_value_26.npy', average_values_array[:, 0])

    plt.ylabel('Observed values',fontsize=16)
    plt.xlabel('Predicted values',fontsize=16)
    plt.title(f'group loss, MAPE={round(group_mape,4)}')
    plt.show()      

    return mape, group_mape


def train_validation(coordinate_3d_list_train, data_list_train, coordinate_3d_list_test, data_list_test,
                            alpha_f_sbi, C_f_ope, epsilon_f_ope, feat_num,  l1_ratio):
    
    ### Same function as above, used only for the 5-fold validation process, return the loss on the validation data
    ### train and save a model (sklearn-elastic net) for each existing training data cluster
    model_parameters_train = []

    for i in range (len(coordinate_3d_list_train)):
        train_data = data_list_train[i]
        train_feature = train_data[:, :feat_num]
        train_efc = train_data[:, 28]

        k_value = 10 if train_feature.shape[1] > 10 else 5 if train_feature.shape[1] > 6 else 3 if train_feature.shape[1] > 3 else 1
        selector_f_regression = SelectKBest(score_func=mutual_info_regression, k=k_value) 
        selector_f_regression.fit(train_feature, train_efc)
        feat_ex = np.argsort(-selector_f_regression.scores_)[(train_feature.shape[0]) - 1 :]
        train_feature[:, feat_ex] = 0
        ############################################################################
        
        f_sbi = linear_model.ElasticNet(alpha = alpha_f_sbi, random_state=0, l1_ratio=l1_ratio)
        f_sbi.fit(train_feature, train_efc)
        coefficients = f_sbi.coef_
        intercept = f_sbi.intercept_  

        parameter = np.insert(coefficients, 0, intercept)
        model_parameters_train.append(parameter)

    f_ope_output = np.array(model_parameters_train)
    f_ope_input = np.array(coordinate_3d_list_train)

    ##### note that here use a SVR model for the f-ope
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    test_coordinate_input = np.array(coordinate_3d_list_test)

    X_train = scaler_X.fit_transform(f_ope_input)
    X_test = scaler_X.transform(test_coordinate_input)
    y_train = scaler_y.fit_transform(f_ope_output)

    svr = SVR(kernel = 'rbf', C = C_f_ope, epsilon = epsilon_f_ope)
    f_ope = MultiOutputRegressor(svr)
    f_ope.fit(X_train, y_train)

    y_pred = f_ope.predict(X_test)
    model_parameters_test = scaler_y.inverse_transform(y_pred)

    test_efc_list = []
    test_pred_list = []
  
    for j in range (len(coordinate_3d_list_test)):

        cur_parameter = model_parameters_test[j, :]
        test_gpr = linear_model.ElasticNet(alpha=1, random_state=0)
        test_gpr.coef_ = cur_parameter[1:]
        test_gpr.intercept_ = cur_parameter[0]

        cur_test_data = data_list_test[j]
        cur_test_sbi = cur_test_data[:, :feat_num]
        cur_test_efc = cur_test_data[:, 28]

        cur_test_pred = np.abs(test_gpr.predict(cur_test_sbi))

        for points_num in range(len(cur_test_pred)):
            test_efc_list.append(cur_test_efc[points_num])
            test_pred_list.append(cur_test_pred[points_num])

    test_efc_list = np.array(test_efc_list)
    test_pred_list = np.array(test_pred_list)
    loss_list = np.abs(test_efc_list - test_pred_list) / test_efc_list
    mape = np.mean(loss_list)
    return mape

# objective function for the k-fold validation process
# this objective is calculated one time for each trial in the grid search, compute the average loss on training split
def objective_linear_k_fold(trial, train_feat_0, train_label_0, train_feat_50, train_label_50):
    
    # define training hyperparameters and their corresponding grid
    alpha_f_sbi    = trial.suggest_float('alpha_f_sbi', 0.8, 1.2, step=0.1)
    C_f_ope        = trial.suggest_float('C_f_ope', 0.9, 1.5, step=0.1)
    epsilon_f_ope  = trial.suggest_float('epsilon_f_ope', 0.1, 0.3, step=0.1)
    l1_ratio       = trial.suggest_float('l1_ratio', 0.4, 0.9, step=0.1)

    coordinate_3d_list_train_efc0_fold, data_list_train_efc0_fold, coordinate_3d_list_test_efc0_fold, data_list_test_efc0_fold = generate_data_list_k_fold(0, 0, train_feat_0, train_label_0)
    coordinate_3d_list_train_efc50_fold, data_list_train_efc50_fold, coordinate_3d_list_test_efc50_fold, data_list_test_efc50_fold = generate_data_list_k_fold(50 ,0, train_feat_50, train_label_50)

    fold_num = len(coordinate_3d_list_train_efc0_fold)
    mape_records = []

    for i in range(fold_num):

        # coordinate_3d_list_train_efc0 = coordinate_3d_list_train_efc0_fold[i]    
        coordinate_3d_list_train_efc50 = coordinate_3d_list_train_efc50_fold[i]   
        # coordinate_3d_list_test_efc0 = coordinate_3d_list_test_efc0_fold[i]    
        coordinate_3d_list_test_efc50 = coordinate_3d_list_test_efc50_fold[i]    

        data_list_train_efc0 = data_list_train_efc0_fold[i]     
        data_list_train_efc50 = data_list_train_efc50_fold[i]    
        data_list_test_efc0 = data_list_test_efc0_fold[i]     
        data_list_test_efc50 = data_list_test_efc50_fold[i]          

        feat_num = 28
        coordinate_3d_list_test = coordinate_3d_list_test_efc50.copy()
        data_list_test = data_list_test_efc50.copy()
        for x in range(len(data_list_test_efc50)):
            data_list_test[x][:, 14:28] = np.abs(data_list_test_efc50[x][:, :14]-data_list_test_efc0[x][:, :14])

        coordinate_3d_list_train = coordinate_3d_list_train_efc50.copy()
        data_list_train = data_list_train_efc50.copy()
        for x in range(len(data_list_train_efc50)):
            data_list_train[x][:, 14:28] = np.abs(data_list_train_efc50[x][:, :14]-data_list_train_efc0[x][:, :14])

        all_feature_test = np.vstack(data_list_test)
        all_feature_train = np.vstack(data_list_train)
        all_feature = np.vstack((all_feature_train[:, :28], all_feature_test[:, :28]))

        mean_all = np.mean(all_feature, axis=0)
        std_all = np.std(all_feature, axis=0, ddof=1)
        for ind in range(len(data_list_train)):
            data_list_train[ind][:, :28] = (data_list_train[ind][:, :28] - mean_all) / std_all
        for ind in range(len(data_list_test)):
            data_list_test[ind][:, :28] = (data_list_test[ind][:, :28] - mean_all) / std_all            

        # delete
        rest_coord = []
        rest_data = []
        delete_v4 = []
        for x in range(len(data_list_train)):
            if data_list_train[x].shape[0] < 7:    ########### delete dataset with points fewer than 1,2,3,4...
                delete_v4.append(x)
                rest_coord.append(coordinate_3d_list_train[x])
                rest_data.append(data_list_train[x])
        sorted_delete_v4 = sorted(delete_v4, reverse=True)
        for index in sorted_delete_v4:
            del coordinate_3d_list_train[index]
            del data_list_train[index]

        mape = train_validation(coordinate_3d_list_train, data_list_train, coordinate_3d_list_test, data_list_test,
                                alpha_f_sbi, C_f_ope, epsilon_f_ope, feat_num,  l1_ratio)
        mape_records.append(mape)
        ave_mape = np.mean(np.array(mape_records))

    return -ave_mape


# testing function, give the final prediction results on the testing data
def objective_linear_test(alpha_f_sbi, C_f_ope, epsilon_f_ope, l1_ratio):
    
    coordinate_3d_list_train_efc0, data_list_train_efc0, coordinate_3d_list_test_efc0, data_list_test_efc0, f1, f2 = generate_data_list_group_mape(0)
    coordinate_3d_list_train_efc50, data_list_train_efc50, coordinate_3d_list_test_efc50, data_list_test_efc50, f1, f2 = generate_data_list_group_mape(50)

    feat_num = 28
    coordinate_3d_list_test = coordinate_3d_list_test_efc50.copy()
    data_list_test = data_list_test_efc50.copy()
    for x in range(len(data_list_test_efc50)):
        data_list_test[x][:, 14:28] = np.abs(data_list_test_efc50[x][:, :14]-data_list_test_efc0[x][:, :14])

    coordinate_3d_list_train = coordinate_3d_list_train_efc50.copy()
    data_list_train = data_list_train_efc50.copy()
    for x in range(len(data_list_train_efc50)):
        data_list_train[x][:, 14:28] = np.abs(data_list_train_efc50[x][:, :14]-data_list_train_efc0[x][:, :14])

    all_feature_test = np.vstack(data_list_test)
    all_feature_train = np.vstack(data_list_train)

    all_feature = np.vstack((all_feature_train[:, :28], all_feature_test[:, :28]))
    mean_all = np.mean(all_feature, axis=0)
    std_all = np.std(all_feature, axis=0, ddof=1)
    for ind in range(len(data_list_train)):
        data_list_train[ind][:, :28] = (data_list_train[ind][:, :28] - mean_all) / std_all
    for ind in range(len(data_list_test)):
        data_list_test[ind][:, :28] = (data_list_test[ind][:, :28] - mean_all) / std_all

    # delete
    rest_coord = []
    rest_data = []
    delete_v4 = []
    for x in range(len(data_list_train)):
        if data_list_train[x].shape[0] < 7:    ########### delete dataset with points fewer than 1,2,3,4...
            delete_v4.append(x)
            rest_coord.append(coordinate_3d_list_train[x])
            rest_data.append(data_list_train[x])
    sorted_delete_v4 = sorted(delete_v4, reverse=True)
    for index in sorted_delete_v4:
        del coordinate_3d_list_train[index]
        del data_list_train[index]

    mape, group_mape = train_test_linear_model(coordinate_3d_list_train, data_list_train, coordinate_3d_list_test, data_list_test,
                            alpha_f_sbi, C_f_ope, epsilon_f_ope, feat_num,  l1_ratio)
    return mape, group_mape


if __name__ == "__main__":
    Testing()


















