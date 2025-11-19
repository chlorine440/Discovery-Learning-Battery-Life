import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression

#22groups
data_train=[
[25,	1.066465897,	0.991850639],
[45,	1.104862773,	0.990403246],
[25,	1.066465897,	0.991850639],
[45,	0.325000006,	0.327970171],
[25,	1.066465897,	0.991850639],
[45,	0.325000006,	0.327970171],
[45,	0.923109751,	1.003950666],
[25,	0.495567899,	1.003951267],
[32.5,	0.393265877,	0.56186204],
[45,	0.936487913,	1.003954133],
[30,	0.319912754,	0.334579284],
[25,	0.357039214,	0.369871081],
[50,	0.934801553,	1.003954129],
[50,	0.934801553,	1.003954129],
[45,	0.335476353,	0.33546792],
[20,	1.187690297,	1.006535969],
[45,	1.167294892,	1.003950965],
[45,	0.910783598,	1.003953749],
[45,	0.959556183,	1.003952304],
[50,	0.945036371,	1.003953973],
[45,	0.945512323,	1.003953866],
[25,	1.033200957,	1.006538975],
]

#22groups
label_train=[676.7422313,
534.3703267,
635.1766627,
463.9636454,
632.8438865,
529.7013091,
441.233851,
1181.114289,
851.3438337,
509.0042233,
977.8483235,
1271.436132,
401.2025135,
382.5410996,
568.4048655,
720.8468085,
492.7555627,
520.2932165,
310.0102607,
394.9669275,
374.8037991,
871.1132565,
]


data_test=[[25,	0.314898257,	0.327664886],
[25,	0.314898257,	0.327664886],
[45,	1.104862773,	0.990403246],
[25,	0.314898257,	0.327664886],
[32.5,	0.393265877,	0.56186204],
[45,	0.945512323,	1.003953866],
[45,	0.923897777,	1.00395415],
[45,	0.936487913,	1.003954133],
[45,	0.936487913,	1.003954133],
[45,	0.959556183,	1.003952304],
[25,	0.314762956,	0.335482699],
[35,	0.880965908,	1.003951568],
[30,	1.018708349,	1.003950972],
[25,	1.419683042,	0.335508155],
[32.5,	0.501249117,	0.557517431],
]

label_test = [1188.00,
1545.00,
506.00,
1112.00,
880.30,
358.00,
464.00,
396,
395.00,
341.00,
879.00,
640.3827279,
696.7314414,
576.3484437,
696.6900333,
]


x_train = np.array(data_train)
y_train = np.array(label_train)
x_test = np.array(data_test)
y_test = np.array(label_test)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) 
kernel = C(1.0, (0.1, 1.5)) * RBF(10, (5, 15))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100,random_state=0)
gp.fit(x_train, y_train)
y_pred, sigma = gp.predict(x_test, return_std=True)

np.set_printoptions(formatter={'float_kind': '{:0.6f}'.format})
# print(y_pred)
print('The standard deviation of the remaining 15 groups:',sigma)
# print(np.abs(y_pred-y_test)/y_test)
# print(np.mean(np.abs(y_pred-y_test)/y_test))

a=sigma
a_mean = np.mean(a)
a_std = np.std(a)
aaa = a_mean-3*a_std
ccc = np.percentile(a, 75)
print('The threshold to determine informative samples (when standard deviation is larger than this value):',ccc)