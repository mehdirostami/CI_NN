
# import numpy as np 


# # print(0.919822*np.sqrt(0.798336-.5)       ,0.938127*np.sqrt(0.924143-.5))
# # print(0.919822*np.sqrt(0.798336-.5)*np.sqrt(1-0.798336)       ,0.938127*np.sqrt(0.924143-.5)*np.sqrt(1-0.924143))


# # print('-------------------')

# # print(0.936926*np.sqrt(0.943372-.5)       ,0.92198*np.sqrt(0.940967-.5))
# # print(0.936926*np.sqrt(0.943372-.5)*np.sqrt(1-0.943372)       ,0.92198*np.sqrt(0.940967-.5)*np.sqrt(1-0.940967))


# # print('-------------------')

# # print(np.sqrt(0.748029-.5)*np.sqrt(1-0.748029)       ,np.sqrt(0.826538-.5)*np.sqrt(1-0.826538))

# # print('-------------------')

# # print(np.sqrt(0.811814-.5)*np.sqrt(1-0.811814)       ,np.sqrt(0.925331-.5)*np.sqrt(1-0.925331))




# # exit()
# import pandas as pd 
# import copy
# import csv
# import argparse

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# from nn_bnn_causality import NNLinear, BNN, NN, justBNN, BNNx

# from simulate_data import SimulateNoIntraction
# from utils import simulate_x, nonlinear, simulate_params, reverse_nonlinear
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVR, SVC

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import r2_score

# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()

# from scipy.optimize import nnls

# import suplearn
# import pylab
# import matplotlib.pyplot as plt
# import time
# import utils
# import training
# import os

# print('-----------------------------')

# import warnings
# warnings.filterwarnings('ignore')
# # import packages needed to run r code in python:



# def Main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("niter", help="Iteration number.", type=int)
#     parser.add_argument("nepoch", help="Number of epochs for NN training.", type=int)
#     parser.add_argument("fold", help="Number of folds for CV.", type=int)
#     parser.add_argument("n", help="Sample size.", type=int)
#     parser.add_argument("p", help="covariate size", type=int)
#     parser.add_argument("rr", help="confounding upper bound", type=float)
#     parser.add_argument("rrrr", help="instrumental var upper bound", type=float)
#     args = parser.parse_args()

#     niter = args.niter
#     nepoch = args.nepoch
#     split_num = args.fold
#     n = args.n
#     p = args.p
#     rr = args.rr
#     rrrr = args.rrrr

#     # print(n, p, rr, rrrr)
#     # print('##########')
#     True_TE = 1

#     if n == 1000:
#         data_dims = [(1000, 10, 10, 10, 10)] 
#     else:
#         data_dims = [(5100, 51, 51, 51, 51)] 

#     if rr == 1.:
#         rr = int(rr)

#     if rrrr == 1.:
#         rrrr = int(rrrr)

#     nonlin_portions = [.2]
#     r_conf = [(.1, rr)]
#     r_iv = [(.1, rrrr)]


    
#     path = "/home/mr/PhD/Causality in AI/Sims2020"

#     for data_dims_ in data_dims:
#         for nonlin_portion in nonlin_portions:
#             for r1, r2 in r_conf:
#                 for r3, r4 in r_iv:

#                     n, p_c, p_iv, p_y, p_n = data_dims_
#                     p = p_c + p_iv + p_y + p_n 

#                     scenario_seed = int(''.join([str(x) for x in [int(n/1000), int(p/100), int(r2), int(r4), niter]]))
#                     seed_beta = int(''.join([str(x) for x in [int(n/1000), int(p/100), int(r2), int(r4)]]))
#                     seed_funcs = seed_beta

#                     params = [niter, n, p, r1, r2, r3, r4]

#                     scenario_name = "dim"+str(n)+"_"+str(p)+'_rConf'+str(r1)+'_'+str(r2)+'_rInstV'+str(r3)+'_'+str(r4)
#                     print('Iteration', niter)
#                     print(scenario_name)
#                     print('-------------------------------------------------------------------')

#                     True_TE = 1
#                     y, A, x, X_c_latent, X_y_latent, ls  = SimulateNoIntraction(True_TE, n, p_c, p_iv, p_y, p_n, rho=.5, corr="AR(1)", nonlinearity_portion=nonlin_portion, r1=r1, r2=r2, r3=r3, r4=r4, sigma=1., seed=scenario_seed, seed_funcs=scenario_seed)
#                     break
#                 break
#             break
#         break

# if __name__ == "__main__":
#     Main()
    
# exit()

import numpy as np 
import pandas as pd 
import copy
import csv
import argparse

# from mlens.ensemble import SuperLearner, Subsemble
import SuperLearner

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error, make_scorer, mean_squared_error, roc_auc_score

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import time
import utils
import os


import warnings
warnings.filterwarnings('ignore')
# import packages needed to run r code in python:




def estimate_supl(data_yAx, n, p, fold, scenario, x_names=None):

    print(x_names)
    print([c for c in data_yAx.columns])
    if x_names is None:
        y, A, x = np.array(data_yAx["y"]).reshape(-1, 1), np.array(data_yAx["A"]).reshape(-1, 1), np.array(data_yAx[["x{}".format(k) for k in range(p)]])
    else:
        y, A, x = np.array(data_yAx["y"]).reshape(-1, 1), np.array(data_yAx["A"]).reshape(-1, 1), np.array(data_yAx[x_names])

    ############################ ############################ ############################ ############################ 
    ############################ ############################ ############################ ############################ 
    ############################ supL
    uuu = importr("tmle")
    ro.r('library("tmle")')

    np2r = np.concatenate([y, A, x], axis=1).T
    np.save("datar{}.npy".format(scenario), np2r)

    _ = ro.r('library(RcppCNPy)')
    _ = ro.r('data_r <- t(npyLoad("datar{}.npy"))'.format(scenario))

    _ = ro.r('y = data_r[, 1]')
    _ = ro.r('A = data_r[, 2]')
    _ = ro.r('W = data_r[, 3:{}]'.format(p+2)) 

    tmle_start = time.clock()
    # _ = ro.r('result <- tmle(Y=y, A=A, W=W, Q.SL.library=c("SL.randomForest", "SL.gam"), g.SL.library=c("SL.randomForest", "SL.gam"), V={})'.format(fold))
    # _ = ro.r('result <- tmle(Y=y, A=A, W=W, Q.SL.library=c("tmle.SL.dbarts2", "SL.glmnet", "SL.gam", "SL.glm"), g.SL.library=c("tmle.SL.dbarts.k.5", "SL.gam", "SL.glm"), V={})'.format(fold))
    _ = ro.r('result <- tmle(Y=y, A=A, W=W, Q.SL.library=c("SL.xgboost", "SL.gam", "SL.glm"), g.SL.library=c("SL.xgboost", "SL.gam", "SL.glm"), V={})'.format(fold))

    tmle_stop = time.clock()
    supl_run_time = np.round(tmle_stop-tmle_start, 2)

    supl_yhat0 = np.array(ro.r('result$Qinit$Q[, "Q0W"]')).reshape(-1, 1)
    supl_yhat1 = np.array(ro.r('result$Qinit$Q[, "Q1W"]')).reshape(-1, 1)
    ps_supl = np.array(ro.r('result$g$g1W')).reshape(-1, 1)
    tmle_est = ro.r("result$estimates$ATE$psi")
    tmle_var = ro.r("result$estimates$ATE$var.psi")
    other_ests = [tmle_est[0], tmle_var[0]]

    supl_y = np.hstack((supl_yhat0, supl_yhat1))
    pred_supl = pd.DataFrame(np.concatenate([supl_y, ps_supl], axis=1))
    # pred_supl.columns = ["supl_yhat0", "supl_yhat1", "ps_supl"] 
    pred_supl.columns = ["supl_yhat0", "supl_yhat1", "ps_supl"] 

    data_yAx = pd.concat([data_yAx, pred_supl], axis=1)

    print(other_ests, supl_run_time)

    coefs_g = np.array(ro.r("result$g$coef")).flatten()
    coefs_Q = np.array(ro.r("result$Qinit$coef")).flatten()

    print("Convex comb. coeffs for g model")
    print(coefs_g)

    print("Convex comb. coeffs for Q model")
    print(coefs_Q)


    y_hat = A*supl_yhat1 + (1-A)*supl_yhat0
    print('R2: ', r2_score(y, y_hat))

    return(data_yAx, other_ests, supl_run_time, coefs_g, coefs_Q)


def estimate_nn_supl(data_yAx, n, p, fold, scenario, x_names=None):

    if x_names is None:
        y, A, x = np.array(data_yAx["y"]).reshape(-1, 1), np.array(data_yAx["A"]).reshape(-1, 1), np.array(data_yAx[["x{}".format(k) for k in range(p)]])
    else:
        y, A, x = np.array(data_yAx["y"]).reshape(-1, 1), np.array(data_yAx["A"]).reshape(-1, 1), np.array(data_yAx[x_names])

    ############################ ############################ ############################ ############################ 
    ############################ ############################ ############################ ############################ 
    ############################ nnet
    uuu = importr("tmle")
    ro.r('library("tmle")')

    np2r = np.concatenate([y, A, x], axis=1).T
    np.save("datar{}.npy".format(scenario), np2r)

    _ = ro.r('library(RcppCNPy)')
    _ = ro.r('data_r <- t(npyLoad("datar{}.npy"))'.format(scenario))

    _ = ro.r('y = data_r[, 1]')
    _ = ro.r('A = data_r[, 2]')
    _ = ro.r('W = data_r[, 3:{}]'.format(p+2))
 
    _ = ro.r('result <- tmle(Y=y, A=A, W=W, Q.SL.library=c("SL.nnet"), g.SL.library=c("SL.nnet"), V={})'.format(fold))

    nnet_yhat0 = np.array(ro.r('result$Qinit$Q[, "Q0W"]')).reshape(-1, 1)
    nnet_yhat1 = np.array(ro.r('result$Qinit$Q[, "Q1W"]')).reshape(-1, 1)
    ps_nnet = np.array(ro.r('result$g$g1W')).reshape(-1, 1)
    tmle_est = ro.r("result$estimates$ATE$psi")
    tmle_var = ro.r("result$estimates$ATE$var.psi")
    other_ests = [tmle_est[0], tmle_var[0]]

    nnet_y = np.hstack((nnet_yhat0, nnet_yhat1))
    pred_nnet = pd.DataFrame(np.concatenate([nnet_y, ps_nnet], axis=1))
    pred_nnet.columns = ["nnet_yhat0", "nnet_yhat1", "ps_nnet"] 

    data_yAx = pd.concat([data_yAx, pred_nnet], axis=1)

    return(data_yAx, other_ests)


def estimate_glm(data_yAx, n, p, fold, scenario, x_names=None):

    # if x_names is None:
    #     y, A, x = np.array(data_yAx["y"]).reshape(-1, 1), np.array(data_yAx["A"]).reshape(-1, 1), np.array(data_yAx[["x{}".format(k) for k in range(p)]])
    # else:
    #     y, A, x = np.array(data_yAx["y"]).reshape(-1, 1), np.array(data_yAx["A"]).reshape(-1, 1), np.array(data_yAx[x_names])
    # if x_names is None:
    #     y, A, x = data_yAx[["y"]], data_yAx[["A"]], data_yAx[["x{}".format(k) for k in range(p)]]
    # else:
    #     y, A, x = data_yAx[["y"]], data_yAx[["A"]], data_yAx[x_names]

    # pd.concat([y, A, x], axis=1).to_csv('temp_data-minic.csv', index=False)
    ############################ ############################ ############################ ############################ 
    ############################ ############################ ############################ ############################ 
    ############################ glm
    uuu = importr("tmle")
    ro.r('library("tmle")')

    # np2r = np.concatenate([y, A, x], axis=1).T
    # np.save("datar{}.npy".format(scenario), np2r)

    # _ = ro.r('library(RcppCNPy)')
    # _ = ro.r('data_r <- npyLoad("datar{}.npy")'.format(scenario))
    _ = ro.r('data_r = read.csv("temp_data-minic.csv")')
    _ = ro.r('y = data_r[, 1]')
    _ = ro.r('A = data_r[, 2]')
    _ = ro.r('W = data_r[, 3:{}]'.format(p+2))
 
    _ = ro.r('result <- tmle(Y=y, A=A, W=W, Q.SL.library=c("SL.glm"), g.SL.library=c("SL.glm"), V={})'.format(fold))

    glm_yhat0 = np.array(ro.r('result$Qinit$Q[, "Q0W"]')).reshape(-1, 1)
    glm_yhat1 = np.array(ro.r('result$Qinit$Q[, "Q1W"]')).reshape(-1, 1)
    ps_glm = np.array(ro.r('result$g$g1W')).reshape(-1, 1)
    tmle_est = ro.r("result$estimates$ATE$psi")
    tmle_var = ro.r("result$estimates$ATE$var.psi")
    other_ests = [tmle_est[0], tmle_var[0]]

    glm_y = np.hstack((glm_yhat0, glm_yhat1))
    pred_glm = pd.DataFrame(np.concatenate([glm_y, ps_glm], axis=1))
    pred_glm.columns = ["glm_yhat0", "glm_yhat1", "ps_glm"] 

    data_yAx = pd.concat([data_yAx, pred_glm], axis=1)

    return(data_yAx, other_ests)




def estimate_tmle_given_preds(data_yAx, n, p, fold, scenario, niter, x_names=None, saved_path=''):

	############################ ############################ ############################ ############################ 
	############################ ############################ ############################ ############################ 
	############################ supL
	uuu = importr("tmle")
	ro.r('library("tmle")')


	file_name = "{}/{}/KfoldQ_ps_supl_oracle{}.csv".format(saved_path, scenario, niter)
	_ = ro.r('data_r = read.csv(file="{}", header = TRUE)'.format(file_name))

	_ = ro.r('y = data_r[, "y"]')
	_ = ro.r('A = data_r[, "A"]')
	_ = ro.r('W = data_r[, paste("x", 0:{}, sep="")]'.format(p-1)) 
	_ = ro.r('Q_0_preds = data_r[, "supl_yhat0"]')
	_ = ro.r('Q_1_preds = data_r[, "supl_yhat1"]')
	_ = ro.r('g_1_pred = data_r[, "ps_supl"]')

	tmle_start = time.clock()
	_ = ro.r('result <- tmle(Y=y, A=A, W=W, Q=cbind(Q_0_preds, Q_1_preds), g1W=g_1_pred, V={})'.format(fold))

	tmle_stop = time.clock()
	supl_run_time = np.round(tmle_stop - tmle_start, 2)
	# print("supl_run_time", supl_run_time)

	tmle_est = ro.r("result$estimates$ATE$psi")
	tmle_var = ro.r("result$estimates$ATE$var.psi")
	other_ests = [tmle_est[0], tmle_var[0]]

	return(other_ests)

