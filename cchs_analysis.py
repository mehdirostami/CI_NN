import numpy as np 
import pandas as pd
import csv, random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib 
from cycler import cycler
from matplotlib.ticker import MaxNLocator
import os
import plotly.express as px

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
import re
import utils
import Rsuperlearner
import training

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dcor

import torch

import importlib
importlib.reload(training)

import importlib
importlib.reload(utils)
from sklearn.linear_model import LogisticRegression , LinearRegression, Lasso


nepoch = 200
split_num1 = 10
split_num2 = 25
np.random.seed(12345)
True_TE = 1


print('device for NN calculations: ', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


path = "/media/mr/DATA/CCHS/"

temp_data = pd.read_csv('/media/mr/DATA/CCHS/cchs_18-65_food_insecure_vs_secure.csv')

n = temp_data.shape[0]



temp_data['split'] = np.floor(np.random.uniform(0, split_num1, size=(n, 1)))

temp_data = temp_data[temp_data['split'].isin([c for c in range(split_num2)])]

temp_data = temp_data.rename({'food_insecure':'A', 'bmi':'y'}, axis=1)
x_names = [c for c in temp_data.columns if c not in ['A', 'y', 'split']]

random.seed(10)


# x_conf = random.sample(x_names, int(398/4))
# x_iv = random.sample([c for c in x_names if c not in x_conf], int(398/4))

# A_covars = x_iv
# y_covars = x_conf + x_ypred

# x_ypred = random.sample([c for c in x_names if c not in A_covars], int(398/4))

# params1 = np.random.choice([-2., 2.], len(A_covars))
# params2 = np.random.choice([-.1, .1], len(y_covars))
        
# xbeta_iv = np.dot(temp_data[A_covars].values, params1)
# pr = 1./(1. + np.exp(-xbeta_iv))
# temp_data['A'] = np.random.binomial(1, pr, size=n)

temp_data[x_names] = StandardScaler().fit_transform(temp_data[x_names])

# X = temp_data[x_names].values
# AX = temp_data[['A'] + x_names].values
# A = temp_data['A'].values.reshape(-1, 1)
# y = temp_data['y'].values.reshape(-1, 1)

# logit = LogisticRegression(fit_intercept=True).fit(X, A)

# params1 = logit.coef_.reshape(-1, 1)

# linreg = Lasso(alpha=.001, fit_intercept=True)

# linreg.fit(AX, y)

# params2 = linreg.coef_.reshape(-1, 1)

# xbeta_iv = np.dot(X, params1)
# pr = 1./(1. + np.exp(-xbeta_iv))

# temp_data['A'] = np.random.binomial(1, pr, size=(n, 1)).reshape(-1, 1)
# xbeta_y = 1 + True_TE * A + np.dot(AX, params2).reshape(-1, 1)
# temp_data['y'] = xbeta_y + np.random.normal(size=(n, 1)) * np.random.normal(size=(n, 1))


n, p = temp_data.shape[0], len(x_names)

print('device for NN calculations: ', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
###################### Running BNN 
hidsize = p
nlayers = 3
nlayersA = 0 # 0 gives BNN, and >=1 gives BNNx
hiddens_list = [[hidsize]*nlayers]
hid2A_list = [[hidsize]*nlayersA]
lasso_list = [1]
decay = 0


x_names = [c for c in temp_data.columns if c not in ['split', 'y', 'A', 'uniq']]

hyperparameters = [
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},
                    ]

for hyp_dict in hyperparameters:

    cv_scenario1 = 'H_{}_L1_{}_L1TG_{}'.format('_'.join([str(int(c)) for c in hyp_dict['hidden']]), hyp_dict['l1_regular'], hyp_dict['l1_targeted'])

    temp_data = training.estimate_targeted_bnn(temp_data, n, p, split_num=split_num2, nepoch=nepoch, 
                          discriminative_regularization=False, hyperparameters=[hyp_dict], 
                          if_partial_linear=True, if_linear=True, architecture='nnY_{}'.format(cv_scenario1), 
                          print_progress=False, get_last_hid_layer=True, x_names=x_names)


hyperparameters = [
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},
    
       
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},
    
    
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},
    
       
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},
        
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},
    
       
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},
        
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},
]
for hyp_dict in hyperparameters:

    cv_scenario2 = 'H_{}_L1_{}_L1TG_{}'.format('_'.join([str(int(c)) for c in hyp_dict['hidden']]), hyp_dict['l1_regular'], hyp_dict['l1_targeted'])

    temp_data = training.estimate_targeted_bnn(temp_data, n, p, split_num=split_num2, nepoch=nepoch, 
                          discriminative_regularization=False, hyperparameters=[hyp_dict], 
                          if_partial_linear=True, if_linear=True, architecture='nnA_{}'.format(cv_scenario2), 
                          print_progress=False, get_last_hid_layer=True, x_names=x_names)

hyperparameters = [
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8, p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .0},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .0},
    
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},
    
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .2, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},

                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p], 'if_linear': True, 'l1_targeted': .1},
                    {'l1_regular': .5, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/8], 'if_linear': True, 'l1_targeted': .1},
                                    ]
                
for hyp_dict in hyperparameters:

    cv_scenario = 'H_{}_L1_{}_L1TG_{}'.format('_'.join([str(int(c)) for c in hyp_dict['hidden']]), hyp_dict['l1_regular'], hyp_dict['l1_targeted'])

    temp_data = training.estimate_targeted_bnn(temp_data, n, p, split_num=split_num2, nepoch=nepoch, 
                          discriminative_regularization=False, hyperparameters=[hyp_dict], 
                          if_partial_linear=True, if_linear=True, architecture='nnAY_{}'.format(cv_scenario), 
                          print_progress=False, get_last_hid_layer=True, x_names=x_names)
                       
                    
    
scen = 'dNN-jNN'
    
temp_data.to_csv('/media/mr/DATA/CCHS/cchs_18-65_food_insecure_vs_secure_pred_{}.csv'.format(scen), index=False)


# temp_data = pd.read_csv('/media/mr/DATA/CCHS/cchs_18-65_food_insecure_vs_secure_pred_{}.csv'.format(scen))

scenarios = set([re.sub('nnAY_|nnY_|nnA_|_yhat1|ps_', '', c) for c in temp_data.columns if 'yhat1' in c or 'ps' in c])
pred_cols = [c for c in temp_data.columns if '_L1_0.' in c]

temp_results1 = []

for fold in temp_data['split'].unique():
    for cv_scenario in scenarios:

        temp_data = temp_data[~temp_data['split'].isna()]

        ##################################
        yhat0 = 'nnY_{}_yhat1'.format(cv_scenario)
        yhat1 = 'nnY_{}_yhat0'.format(cv_scenario)
        ps = 'ps_nnA_{}'.format(cv_scenario)

        temp_results_ = training.save_ests(temp_data[temp_data['split']==fold], p, params=[], params_names=[], 
                                            methods=[{yhat0, yhat1, ps}], path='', output_name='', x_names = x_names)

        temp_results_.columns = [re.sub('_2nns_ps|_nnY|_'+cv_scenario, '', c) for c in temp_results_.columns]
        temp_results_.columns = [re.sub('_2nns', '', c) for c in temp_results_.columns]

        hid = re.sub("'","",str(re.search('H_(.+)L1_', cv_scenario).group()[2:][:-4].split('_')))

        temp_results_['hidden'] = hid
        temp_results_['L1'] = re.search('L1_(.+)_L1', cv_scenario).group()[3:][:-3]
        temp_results_['L1TG'] = re.search('L1TG_(.+)', cv_scenario).group()[5:]
        temp_results_['cv_scenario'] = cv_scenario
        temp_results_['architecture'] = 'dNN'
        temp_results_['fold'] = fold
        
        temp_results1 += [temp_results_]

        ####################################
        yhat0 = 'nnAY_{}_yhat1'.format(cv_scenario)
        yhat1 = 'nnAY_{}_yhat0'.format(cv_scenario)
        ps = 'ps_nnAY_{}'.format(cv_scenario)

        temp_results = training.save_ests(temp_data[temp_data['split']==fold], p, params=[], params_names=[], 
                                    methods=[{yhat0, yhat1, ps}], path='', output_name='', x_names = x_names)

        temp_results.columns = [re.sub('_nnAY|_'+cv_scenario, '', c) for c in temp_results.columns]

        temp_results['hidden'] = hid
        temp_results['L1'] = re.search('L1_(.+)_L1', cv_scenario).group()[3:][:-3]
        temp_results['L1TG'] = re.search('L1TG_(.+)', cv_scenario).group()[5:]
        temp_results['cv_scenario'] = cv_scenario
        temp_results['architecture'] = 'jNN'
        temp_results['fold'] = fold

        temp_results1 += [temp_results]
            
            
results_df = pd.concat(temp_results1, axis=0)
print(results_df['cv_scenario'].value_counts())

print('results_df--done')
results_df.to_csv("/media/mr/DATA/CCHS/dNN-cchs-results-Dec2021_{}.csv".format(scen), index=False)




