
import numpy as np 
import pandas as pd 
import copy
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from nn_bnn_causality import NNLinear, BNN, NN, justBNN, BNNx

from simulate_data import SimulateNoIntraction
from utils import simulate_x, nonlinear, simulate_params, reverse_nonlinear
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from scipy.optimize import nnls

# import suplearn
import pylab
import matplotlib.pyplot as plt
import time
import utils
import training
import os
import statsmodels.api as sm


import warnings
warnings.filterwarnings('ignore')
# import packages needed to run r code in python:


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("niter", help="Iteration number.", type=int)
    parser.add_argument("nepoch", help="Number of epochs for NN training.", type=int)
    parser.add_argument("fold", help="Number of folds for CV.", type=int)
    parser.add_argument("n", help="Sample size.", type=int)
    parser.add_argument("p", help="covariate size", type=int)
    parser.add_argument("lb", help="lower bound of adjusting factors' effects", type=float)
    parser.add_argument("ub", help="upper bound of adjusting factors' effects", type=float)
    args = parser.parse_args()

    niter = args.niter
    nepoch = args.nepoch
    split_num = args.fold
    n = args.n
    p = args.p
    lb = args.lb
    ub = args.ub

    True_TE = 1
    
    data_dims = [(n, int(p/4.), int(p/4.), int(p/4.), int(p/4.))]

    nonlin_portions = [0]
    r = [(lb, ub)]

    path = "/home/mr/PhD/Causality in AI/Sim2022"
    # save_path = '/home/mr/PhD/Causality in AI/Sim2022'

    seeds = []

    for data_dims_ in data_dims:
        for nonlin_portion in nonlin_portions:
            for r1, r2 in r:

                n, p_c, p_iv, p_y, p_n = data_dims_
                p = p_c + p_iv + p_y + p_n 

                scenario_seed = int(''.join([str(x) for x in [int(n/10000), int(p/100.), int(int(10*r1)/10.), int(10*r2), niter]]))
                seed_beta = int(''.join([str(x) for x in [int(n/10000), int(p/100.), int(int(10*r1)/10.), int(10*r2)]]))
                seed_funcs = seed_beta

                params = [niter, n, p, r1, r2]
                scenario_name = "featureImp_dim{}_{}_r_{}_{}".format(str(n), str(p), str(r1), str(r2))
                print('Iteration', niter)
                print(scenario_name)
                print('-------------------------------------------------------------------')

                # temp_data = pd.read_csv("{}/{}/KfoldQ_ps_nns{}.csv".format(path, scenario_name, niter))

                True_TE = 1
                y, A, x, x_in_DGP  = SimulateNoIntraction(True_TE, n, p_c, p_iv, p_y, p_n, rho=.5, corr="AR(1)", 
                                                            nonlinearity_portion=nonlin_portion, dist='normal', 
                                                            r1=0, r2=0, r3=r1, r4=r2, sigma=1., seed=scenario_seed, 
                                                            seed_funcs=scenario_seed, plot=False
                                                          )

                # print((y*A/(A.sum())).sum()-(y*(1-A)/((1-A).sum())).sum())
                # exit()
                # print(list(x_in_DGP.keys()))
                
                # ['X_c_latent', 'X_iv_latent', 'X_y_latent']
                confx = x_in_DGP['X_c_latent']
                ypredx = x_in_DGP['X_y_latent']
                iv_predx = x_in_DGP['X_iv_latent']

                x_c_A_y = np.hstack((np.hstack((A, confx)), ypredx))
                x_c_A = np.hstack((A, confx))
                x_c_A0 = np.hstack((np.hstack((np.zeros((A.shape[0], 1)), confx)), ypredx))
                x_c_A1 = np.hstack((np.hstack((np.ones((A.shape[0], 1)), confx)), ypredx))
                x_c_iv = np.hstack((confx, iv_predx))

                temp_oracle_Amodel = LogisticRegression(fit_intercept=True).fit(x, A)

                temp_oracle_ymodel = Ridge(alpha=0.000001, fit_intercept=True).fit(x_c_A_y, y)

                # xcols = ["x{}".format(k) for k in range(p)]
                # y_df = pd.DataFrame(y.reshape(-1, 1), columns=['y'])
                # A_df = pd.DataFrame(A.reshape(-1, 1), columns=['A'])
                # x_df = pd.DataFrame(x, columns=xcols)

                split = np.floor(np.random.uniform(0, split_num, size=(n, 1)))
                temp_data = pd.DataFrame(np.concatenate([split, y, A, x], axis=1))
                temp_data.columns = ["split"] + ["y", "A"] + ["x{}".format(k) for k in range(p)]
                temp_data = temp_data.sort_values("split")
                temp_data = temp_data.reset_index()
                temp_data = temp_data.rename({'index':'id'})

                temp_data["ps_oracle"] = temp_oracle_Amodel.predict_proba(x)[:, 1].flatten()

                temp_data["oracle_yhat0"] = temp_oracle_ymodel.predict(x_c_A0).flatten()
                temp_data["oracle_yhat1"] = temp_oracle_ymodel.predict(x_c_A1).flatten()

                temp_data.to_csv(f"{path}/{scenario_name}/KfoldQ_ps_nns{niter}.csv")


                exit()
                
                ###################### Running BNN 
                hidsize = p
                nlayers = 3
                nlayersA = 0 # 0 gives BNN, and >=1 gives BNNx
                hiddens_list = [[hidsize]*nlayers]
                hid2A_list = [[hidsize]*nlayersA]
                lasso_list = [1]
                decay = 0

                hyperparameters = [
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .0},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .0},


                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .1},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .1},


                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .3},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .3},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .3},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .3},


                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .7},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .7},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .7},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .7},

                                    ]
                
                for hyp_dict in hyperparameters:

                    cv_scenario = 'H_{}_L1_{}_L1TG_{}'.format('_'.join([str(int(c)) for c in hyp_dict['hidden']]), hyp_dict['l1_regular'], hyp_dict['l1_targeted'])
                    print('----------------------------------nnAY1---------------------------------')
                    
                    temp_data = training.estimate_targeted_bnn(temp_data, n, p, split_num=split_num, nepoch=nepoch, 
                                          discriminative_regularization=False, hyperparameters=[hyp_dict], 
                                          if_partial_linear=True, if_linear=True, architecture='nnAY_{}'.format(cv_scenario), 
                                          print_progress=False, get_last_hid_layer=True)
                    print('cross-validated hyperparameters', cv_scenario)


                

                hyperparameters = [
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .0},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .0},


                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .1},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .1},

                                    
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .3},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .3},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .3},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .3},
                                    
                                    
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .7},
                                    {'l1_regular': .01, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .7},

                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .7},
                                    {'l1_regular': .1, 'yloss_weight': 1, 'Aloss_weight': 0, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .7},

                                    
                                    ]

                for hyp_dict in hyperparameters:

                    cv_scenario1 = 'H_{}_L1_{}_L1TG_{}'.format('_'.join([str(int(c)) for c in hyp_dict['hidden']]), hyp_dict['l1_regular'], hyp_dict['l1_targeted'])
                    print('----------------------------------nnY---------------------------------')
                    
                    temp_data = training.estimate_targeted_bnn(temp_data, n, p, split_num=split_num, nepoch=nepoch, 
                                          discriminative_regularization=False, hyperparameters=[hyp_dict], 
                                          if_partial_linear=True, if_linear=True, architecture='nnY_{}'.format(cv_scenario1), 
                                          print_progress=False, get_last_hid_layer=True)
                    print('cross-validated hyperparameters', cv_scenario1)


                hyperparameters = [
                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .0},

                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .0},
                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .0},


                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .1},

                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .1},
                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .1},


                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .3},
                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .3},

                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .3},
                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .3},


                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .7},
                                    {'l1_regular': .01, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .7},

                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p, p, p], 'if_linear': True, 'l1_targeted': .7},
                                    {'l1_regular': .1, 'yloss_weight': 0, 'Aloss_weight': 1, 'DisReg_weight': 0, 'hidden': [p/10., p, p/10.], 'if_linear': True, 'l1_targeted': .7},

                                    ]
                for hyp_dict in hyperparameters:

                    cv_scenario2 = 'H_{}_L1_{}_L1TG_{}'.format('_'.join([str(int(c)) for c in hyp_dict['hidden']]), hyp_dict['l1_regular'], hyp_dict['l1_targeted'])
                    print('----------------------------------nnA---------------------------------')
                    
                    temp_data = training.estimate_targeted_bnn(temp_data, n, p, split_num=split_num, nepoch=nepoch, 
                                          discriminative_regularization=False, hyperparameters=[hyp_dict], 
                                          if_partial_linear=True, if_linear=True, architecture='nnA_{}'.format(cv_scenario2), 
                                          print_progress=False, get_last_hid_layer=True)
                    print('cross-validated hyperparameters', cv_scenario2)

                
                temp_data.to_csv("{}/{}/KfoldQ_ps_nns{}.csv".format(path, scenario_name, niter), index=False)

if __name__ == "__main__":
    Main()

