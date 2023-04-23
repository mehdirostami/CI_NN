
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from numpy.random import normal, multivariate_normal
from cvxopt import matrix, solvers
import warnings
from sklearn.cluster import AgglomerativeClustering
import multiprocessing 
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from nn_bnn_causality import NNLinear, BNN, NN, justBNN, BNNx

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import tabulate
# tabulate.LATEX_ESC
import re
APE_RULES={}


def to_latex_table(df, num_header=2, remove_char=[]):

    if num_header == 2:
        df2 = df.reset_index().T.reset_index().T
    else:
        df2 = df.T.reset_index().T
    remove_cols = ['0', 'index']
    # print(df2)
    # for col in remove_cols:
    #     if col in list(df2):
    #         df2 = df2.drop(col, axis=1)
    #         print(col)
    
    out = tabulate.tabulate(df2.reset_index().iloc[:, 2:], tablefmt='latex_raw', showindex='never')
    out = out.replace("\\\\", "\\")
    out = out.replace("\\\\", "\\")
    out = out.replace("\\\n", "\\\\ \\hline\n")
    out = re.sub(' +', ' ', out)

    out = out.replace("est" , "Estimator")
    out = out.replace("pred" , "Prediction")
    out = out.replace("means" , "Mean")
    out = out.replace("bias" , "Bias")
    out = out.replace("std" , "MC std")
    out = out.replace("MC stdapprox" , "Est. std")
    out = out.replace("MCerror" , "MC error")

    out = out.replace("rob", "Robinson")
    out = out.replace("Nipw", "nIPW")

    out = out.replace("lllllllllll", "llccccccccc")

    out = out.replace("color & ", "")
    out = out.replace("\\rowcolor{Gray} &", "\\rowcolor{Gray}")
    out = out.replace("\\rowcolor{LightCyan} &", "\\rowcolor{LightCyan}")
    out = out.replace("\\rowcolor{Yellow} &", "\\rowcolor{Yellow}")

    for char in remove_char:
        out = out.replace(char, "")

    # out = eval(out)
    return(out)



def SR(y, A, x, Q1, Q0, last_hidden=np.array([])):

    y_hat = A * Q1 + (1 - A) * Q0

    residual = y - y_hat
    n, p = x.shape

    ## H = last_hidden
    # last_hidden = np.array([])

    # if len(last_hidden) != 0:
    #     k = 2 + last_hidden.shape[1]
    #     HtH_inv = np.linalg.inv(np.dot(H.T, H))

    #     AtH = np.dot(A.T, H)
    #     HtA = AtH.T
        
    #     projection = np.dot(AtH, np.dot(HtH_inv, HtA))
    #     correction = np.sum(A) - projection
    #     correction = correction[0, 0]
    #     if correction < 0.:
    #         raise ValueError("Correction term shouldn't be negative")
    # elif len(last_hidden) == 0:
    #     k = 2 + x.shape[1]
    #     correction = np.sum(A) - k
        
    # if correction == 0.:
    #     correction = 1.

    est, var = (Q1-Q0).mean(), np.sum((residual) ** 2)/(n**2)
    return(est, var)

def naiveATE(y, A):
    return(np.mean(A*y) - np.mean((1-A)*y))

def ipw(y, A, W, ps):

    n = y.shape[0]

    y = y.reshape(-1, 1)
    A = A.reshape(-1, 1)
    ps = ps.reshape(-1, 1)
    w1 = (1. / ps).reshape(-1, 1)
    w0 = (1. / (1. - ps)).reshape(-1, 1)
    
    r1 = (y * w1 * A)/n
    r0 = (y * w0 * (1 - A))/n
    
    n = y.shape[0]
    
    mu1, mu0 = np.sum(r1), np.sum(r0)
    est = mu1 - mu0
    
    I = ((A * y * w1) - ((1-A) * y * w0).reshape(-1, 1)) - est  
    var_crude = np.dot(I.T, I)/n**2
    
    return(est, var_crude[0, 0])

def normalized_ipw(y, A, W, ps):

    y = y.reshape(-1, 1)
    A = A.reshape(-1, 1)
    ps = ps.reshape(-1, 1)
    w1 = (1. / ps).reshape(-1, 1)
    w0 = (1. / (1. - ps)).reshape(-1, 1)
    
    r1 = (y * w1 * A)/(np.sum(w1 * A))
    r0 = (y * w0 * (1 - A))/(np.sum(w0 * (1 - A)))
    
    n = y.shape[0]
    
    mu1, mu0 = np.sum(r1), np.sum(r0)

    est = mu1 - mu0

    I = (A * (y-mu1) * w1) - ((1-A) * (y-mu0) * w0).reshape(-1, 1)    
    var_crude = np.dot(I.T, I)/n**2

    return(est, var_crude[0, 0])

def DR(y, A, ps, Q0, Q1):
    n = y.shape[0]
    
    w1 = 1. / ps 
    w0 = 1. / (1 - ps)

    r1 = (A * y - Q1 * (A - ps)) * w1
    r0 = ((1 - A) * y + Q0 * (A - ps)) * w0
    
    est = np.mean(r1) - np.mean(r0)
    I1 = (((A * y - Q1 * (A-ps)) * w1 - ((1-A) * y + Q0 * (A-ps))* w0) - est).reshape(-1, 1)
    var = np.dot(I1.T, I1)/n**2
    
    return(est, var[0, 0])


def nDR(y, A, ps, Q0, Q1):

    n = y.shape[0]
    w1 = 1. / ps 
    w0 = 1. / (1 - ps)

    r1 = A * (y - Q1) * w1
    r0 = (1 - A) * (y - Q0) * w0
    
    s1 = np.sum(r1)/np.sum(A * w1)
    s0 = np.sum(r0)/np.sum((1-A) * w0)
    
    SR_est = Q1.mean()-Q0.mean()
    nDR_est = (s1 - s0) + SR_est

    # I = (((A * (y - Q1)) * w1/np.sum(A * w1) - ((1-A) * (y - Q0) )* w0/np.sum((1-A) * w0)) + (SR_est - nDR_est)/n).reshape(-1, 1)    
    I = (((A * (y - Q1)) * w1/np.sum(A * w1) - ((1-A) * (y - Q0) )* w0/np.sum((1-A) * w0)) + (SR_est - nDR_est)/n).reshape(-1, 1)    
    var = np.dot(I.T, I)
    return(nDR_est, var[0, 0])


def TailnDR(y, A, ps, Q0, Q1):

    n = y.shape[0]

    w1 = 1. / ps 
    w0 = 1. / (1 - ps)

    normalization1 = np.mean(A * w1)
    normalization0 = np.mean((1-A) * w0)

    # cutoff1 = np.quantile(ps, .01)
    # cutoff0 = np.quantile(ps, .99)

    cutoff1 = np.quantile(ps, 1/n)
    cutoff0 = np.quantile(ps, 1-1/n)

    truncated_ps_indicator_A1 = 0. + (ps > cutoff1)
    truncated_ps_indicator_A0 = 0. + (ps < cutoff0)

    mixed_ps1 = ps * (normalization1 * (1-truncated_ps_indicator_A1) + truncated_ps_indicator_A1)
    mixed_ps0 = (1-ps) * (normalization0 * (1-truncated_ps_indicator_A0) + truncated_ps_indicator_A0)

    w1 = 1. / mixed_ps1 
    w0 = 1. / mixed_ps0

    r1 = (A * y - Q1 * (A - ps)) * w1
    r0 = ((1 - A) * y + Q0 * (A - ps)) * w0
    
    est = np.mean(r1) - np.mean(r0)
    I1 = (((A * y - Q1 * (A-ps)) * w1 - ((1-A) * y + Q0 * (A-ps))* w0) - est).reshape(-1, 1)
    var = np.dot(I1.T, I1)/n**2

    return(est, var[0, 0])



def robinson(y, A, ps, Q0, SR):
    y, A, ps, Q0 = y.reshape(-1, 1), A.reshape(-1, 1), ps.reshape(-1, 1), Q0.reshape(-1, 1)

    V = A - ps
    n = y.shape[0]        

    W = y - Q0
    beta = np.dot(V.T, W)/np.dot(V.T, V)
    
    V2 = V * V
    W2 = W * W
    var = (V2 * W2).mean()/(n * (V2.mean())**2)
    
    return(beta[0, 0], var)

def robinson2(y, A, ps, Q0, SR):

    y, A, ps, Q0 = y.reshape(-1, 1), A.reshape(-1, 1), ps.reshape(-1, 1), Q0.reshape(-1, 1)
        
    V = A - ps
    n = y.shape[0]

    W = y - Q0
    beta = np.dot(V.T, W)/np.dot(A.T, V)
    
    V2 = V * A
    W2 = W * W
    var = (V2 * W2).mean()/(n * ((V2).mean())**2)

    return(beta[0, 0], var)

def CausalEstimators(y, A, W, Q1, Q0, ps, last_hidden=dict()):

    # SR
    SR_est = SR(y, A, W, Q1, Q0, last_hidden)
    # normalized_ipw
    ipw_est = ipw(y, A, W, ps)
    normalized_ipw_est = normalized_ipw(y, A, W, ps)
    # DR
    DR_est = DR(y, A, ps, Q0, Q1)
    # nDR
    nDR_est = nDR(y, A, ps, Q0, Q1)
    # nDR
    TailnDR_est = TailnDR(y, A, ps, Q0, Q1)

    # robinson
    robinson_est = robinson(y, A, ps, Q0, SR_est)
    # robinson2
    robinson2_est = robinson2(y, A, ps, Q0, SR_est)

    est_dict = {'SR': SR_est, 
                'ipw': ipw_est, 
                'normalized_ipw': normalized_ipw_est, 
                'DR': DR_est, 
                'nDR': nDR_est, 
                'TailnDR': TailnDR_est, 
                'robinson': robinson_est, 
                'robinson2': robinson2_est
                }

    return(est_dict)


def PredictionMeasures(y, A, yhat1, yhat0, ps):
    y_hat = (1-A) * yhat0 + A* yhat1

    # R2 bnn
    r2_score_est = metrics.r2_score(y, y_hat)

    # AUC ps bnn
    roc_auc_score_est = metrics.roc_auc_score(A, ps)

    # Brier ps bnn
    brier_est = brier(A, ps)

    # psudo-R2 ps bnn
    psudor2_est = psudor2(A, ps)

    mse = ((y_hat-y)**2).mean()
    CE = (A * np.log(ps) + (1 - A) * np.log(1 - ps)).sum()

    measures_dict = {"r2_score":r2_score_est, "roc_auc_score":roc_auc_score_est, "brier":brier_est, "psudor2":psudor2_est, 'mse':mse, 'CE': CE}
    return(measures_dict)


def print_ests(y, A, W, yhat1, yhat0, ps):

    ests = CausalEstimators(y, A, W, yhat1, yhat0, ps)
    print()
    print('SR: ')
    print(ests["SR"])
    print()
    print('normalized_ipw: ')
    print(ests["normalized_ipw"])
    print()
    print('DR: ')
    print(ests["DR"])
    print()
    print('nDR: ')
    print(ests["nDR"])
    print()
    print('robinson: ')
    print(ests["robinson"])
    print()
    print('robinson2: ')
    print(ests["robinson2"])
    print()

    measures = PredictionMeasures(y, A, yhat1, yhat0, ps)
    print("R2 ")
    print(measures['r2_score'])

    print()
    print("AUC ps ")
    print(measures['roc_auc_score'])

    print()
    print("Brier ps ")
    print(measures['brier'])

    print()
    print("psudo-R2 ps ")
    print(measures['psudor2'])


# def bnnSupL(A, y, list_yhat1, list_yhat0, list_ps):

#     X_ps = np.concatenate(list_ps, axis = 1)
#     X_y1 = np.concatenate(list_yhat1, axis = 1)
#     X_y0 = np.concatenate(list_yhat0, axis = 1)

#     list_yhat = [A * bnny1 + (1-A) * bnny0 for bnny1, bnny0 in zip(list_yhat1, list_yhat0)]
#     X_y = np.concatenate(list_yhat, axis = 1)

#     lin_ps = Lasso(alpha=0.00001, precompute=True, max_iter=1000, positive=True, random_state=9999, selection='random', fit_intercept=False)
#     lin_ps.fit(X_ps, A)
#     supps_coef = lin_ps.coef_.reshape(-1, 1)
#     supps_coef = supps_coef/supps_coef.sum()

#     lin_y = Lasso(alpha=0.00001, precompute=True, max_iter=1000, positive=True, random_state=5555, selection='random', fit_intercept=False)
#     lin_y.fit(X_y, y)
#     supy_coef = lin_y.coef_.reshape(-1, 1)
#     supy_coef = supy_coef/supy_coef.sum()

#     supps = np.dot(X_ps, supps_coef)
#     supy1 = np.dot(X_y1, supy_coef)
#     supy0 = np.dot(X_y0, supy_coef)

#     return(supps_coef, supy_coef, supy1, supy0, supps)



# def bnnJacob(A, x, split, trained_models_dict, device):

#     output_jacob = 0. * x
#     treatment_jacob = 0. * x

#     for fold, model in trained_models_dict.items():

#         i = int(fold)

#         A_tensor_ = Variable(torch.from_numpy(A[np.where(split == i)[0]].reshape(-1, 1)).float(), requires_grad = True).to(device)
#         x_tensor_ = Variable(torch.from_numpy(x[np.where(split == i)[0], :]).float(), requires_grad = True).to(device)

#         model.zero_grad()
#         A_tensor_.retain_grad()
#         outputy, outputA  = model(x_tensor_, A_tensor_)

#         output_jacob[np.where(split == i)[0], :] = torch.autograd.grad(torch.sum(outputy), x_tensor_, retain_graph=True)[0].to('cpu').numpy()
#         treatment_jacob[np.where(split == i)[0]] = torch.autograd.grad(torch.sum(outputA), x_tensor_, retain_graph=True)[0].to('cpu').numpy()


#     return(output_jacob, treatment_jacob)

def get_last_hidden_lay(A, x, split, trained_models_dict, device, hidden):

    layer = np.zeros((x.shape[0], hidden[-1]))

    for fold, model in trained_models_dict.items():

        i = int(fold)

        A_tensor_ = Variable(torch.from_numpy(A[np.where(split == i)[0]].reshape(-1, 1)).float(), requires_grad = True).to(device)
        x_tensor_ = Variable(torch.from_numpy(x[np.where(split == i)[0], :]).float(), requires_grad = True).to(device)

        model.zero_grad()
        layer[np.where(split == i)[0], :]  = model.get_last_hidden_layer(x_tensor_, A_tensor_).to("cpu").data.numpy()
    
    return(layer)


def bnnJacob(A, x, split, trained_models_dict, device):

    output_jacob_dict = {}
    treatment_jacob_dict = {}

    for fold, model in trained_models_dict.items():

        i = int(fold)

        A_tensor_ = Variable(torch.from_numpy(A[np.where(split == i)[0]].reshape(-1, 1)).float(), requires_grad = True).to(device)
        x_tensor_ = Variable(torch.from_numpy(x[np.where(split == i)[0], :]).float(), requires_grad = True).to(device)

        model.zero_grad()
        A_tensor_.retain_grad()
        outputy, outputA  = model(x_tensor_, A_tensor_)

        output_jacob_dict[i] = torch.autograd.grad(torch.sum(outputy), x_tensor_, retain_graph=True)[0].to('cpu').numpy()
        treatment_jacob_dict[i] = torch.autograd.grad(torch.sum(outputA), x_tensor_, retain_graph=True)[0].to('cpu').numpy()

    return(output_jacob_dict, treatment_jacob_dict)



def plnnJacob(A, x, split, trained_models_dict, device):

    output_jacob = 0. * x

    for fold, model in trained_models_dict.items():

        i = int(fold)

        A_tensor_ = Variable(torch.from_numpy(A[np.where(split == i)[0]].reshape(-1, 1)).float(), requires_grad = True).to(device)
        x_tensor_ = Variable(torch.from_numpy(x[np.where(split == i)[0], :]).float(), requires_grad = True).to(device)

        model.zero_grad()
        A_tensor_.retain_grad()
        outputy  = model(x_tensor_, A_tensor_)

        output_jacob[np.where(split == i)[0], :] = torch.autograd.grad(torch.sum(outputy), x_tensor_, retain_graph=True)[0].to('cpu').numpy()

    return(output_jacob)



def sqVar(jacob_vac):
    # A_VI_mean = np.mean(jacob_vac, axis=0)
    # A_VI_meanabs = np.mean(np.abs(jacob_vac), axis=0
    return(np.sqrt(np.var(jacob_vac, axis=0)))

def count_captured_largest_effects(true_index, vector):

    q = len(true_index)
    vector_sort = np.argsort(vector)[::-1][:q] # index of p/2 largest importance variables

    num_of_captured_in_true_index = sum([index in true_index for index in vector_sort])
    return(num_of_captured_in_true_index)


# def count_captured_largest_effects_Apred(true_index, A_VI_sqvar):

#     q = len(true_index)
#     A_VI_meanabs_sort = np.argsort(A_VI_meanabs)[::-1][:q] # index of p/2 largest importance variables
#     num_of_captured_conf_ypred = sum([index in true_index for index in A_VI_meanabs_sort])
#     print(num_of_captured_conf_ypred)


def print_VI(explanatory_VI_mean, explanatory_VI_meanabs, explanatory_VI_sqvar, scale = 10e8):

    p = explanatory_VI_mean.shape[0]
    print('--------------------------------------------------------------------------------------------------')
    print("naiveATE effects", np.round(scale * explanatory_VI_mean[:int(p/4)], 2))
    print("Instrumental variable effects", np.round(scale * explanatory_VI_mean[int(p/4):(2*int(p/4))], 2))
    print("Outcome predictor effects", np.round(scale * explanatory_VI_mean[(2*int(p/4)):(3*int(p/4))], 2))
    print("Irrelevant variable effects", np.round(scale * explanatory_VI_mean[(3*int(p/4)):(4*int(p/4))], 2))
    print('--------------------------------------------------------------------------------------------------')
    print("naiveATE effects", np.round(scale * explanatory_VI_meanabs[:int(p/4)], 2))
    print("Instrumental variable effects", np.round(scale * explanatory_VI_meanabs[int(p/4):(2*int(p/4))], 2))
    print("Outcome predictor effects", np.round(scale * explanatory_VI_meanabs[(2*int(p/4)):(3*int(p/4))], 2))
    print("Irrelevant variable effects", np.round(scale * explanatory_VI_meanabs[(3*int(p/4)):(4*int(p/4))], 2))
    print('--------------------------------------------------------------------------------------------------')
    print("naiveATE effects", np.round(scale * explanatory_VI_sqvar[:int(p/4)], 2))
    print("Instrumental variable effects", np.round(scale * explanatory_VI_sqvar[int(p/4):(2*int(p/4))], 2))
    print("Outcome predictor effects", np.round(scale * explanatory_VI_sqvar[(2*int(p/4)):(3*int(p/4))], 2))
    print("Irrelevant variable effects", np.round(scale * explanatory_VI_sqvar[(3*int(p/4)):(4*int(p/4))], 2))


def ConfImp(y_VI, A_VI=None, method="prod_var"):
    """
    naiveATE importance
    y_VI and A_VI are two matrices of size n*p.
    """
    n, p = y_VI.shape
    if A_VI is not None:
        if y_VI.shape != A_VI.shape:
            raise ValueError("y_VI and A_VI should have the same size.")

    VIs = []
    if A_VI is not None:
        for j in range(p):
            if method.lower() in ["covar", "covariance", 'cov']:
                VIs += [np.cov(np.column_stack((y_VI[:, j], A_VI[:, j])), rowvar=False)[0, 1]]
            elif method.lower() in ["var_prod"]:
                VIs += [np.sqrt(np.var(y_VI[:, j] * A_VI[:, j]))]
            elif method.lower() in ["prod_var"]:
                VIs += [np.std(A_VI[:, j]) * np.std(y_VI[:, j])]
    else:
        VIs = np.std(y_VI, axis=1)

    return(np.array(VIs))


def largest_contributions(y_VI_sqvar, A_VI_sqvar=False, confounder_effect=False, first_set = 50):

    conf_index = [j for j in range(first_set)] # conf_index = [j for j in range(int(p/4))]
    ypred_index = [j for j in range((2*first_set), (3*first_set))] #     ypred_index = [j for j in range((2*int(p/4)), (3*int(p/4)))]
    iv_index = [j for j in range((1*first_set), (2*first_set))] #     iv_index = [j for j in range((1*int(p/4)), (2*int(p/4)))]
    irr_index = [j for j in range((3*first_set), (4*first_set))] #     irr_index = [j for j in range((3*int(p/4)), (4*int(p/4)))]

    contribute_dict = {'conf':dict(), 'ypred':dict(), 'iv':dict(), 'irr':dict()}

    contribute_dict['conf']['y_VI'] = count_captured_largest_effects(conf_index, y_VI_sqvar)
    contribute_dict['ypred']['y_VI'] = count_captured_largest_effects(ypred_index, y_VI_sqvar)
    contribute_dict['iv']['y_VI'] = count_captured_largest_effects(iv_index, y_VI_sqvar)
    contribute_dict['irr']['y_VI'] = count_captured_largest_effects(irr_index, y_VI_sqvar)

    if A_VI_sqvar is not False:

        contribute_dict['conf']['ConfImp'] = count_captured_largest_effects(conf_index, confounder_effect)
        contribute_dict['ypred']['ConfImp'] = count_captured_largest_effects(ypred_index, confounder_effect)
        contribute_dict['iv']['ConfImp'] = count_captured_largest_effects(iv_index, confounder_effect)
        contribute_dict['irr']['ConfImp'] = count_captured_largest_effects(irr_index, confounder_effect)

        contribute_dict['conf']['A_VI'] = count_captured_largest_effects(conf_index, A_VI_sqvar)
        contribute_dict['iv']['A_VI'] = count_captured_largest_effects(iv_index, A_VI_sqvar)
        contribute_dict['ypred']['A_VI'] = count_captured_largest_effects(ypred_index, A_VI_sqvar)
        contribute_dict['irr']['A_VI'] = count_captured_largest_effects(irr_index, A_VI_sqvar) 

    return(contribute_dict)



def print_largest_contributions(confounder_effect, y_VI_sqvar, A_VI_sqvar=False, first_set = 50):

    if A_VI_sqvar is not False:
        nums = largest_contributions(confounder_effect, y_VI_sqvar, A_VI_sqvar, first_set)
    elif A_VI_sqvar is False:
        nums = largest_contributions(confounder_effect, y_VI_sqvar, first_set)

    for contrib_name, contrib in nums.items():
        for importance_name, importance in contrib.items():
            print("How many of the first {} of {} measures are among {} covariates: ".format(first_set, importance_name, contrib_name))
            print(nums[contrib_name][importance_name])


def brier(obs_binary, pred_prob):
    return(((pred_prob - obs_binary)**2).mean())


def psudor2(obs_binary, pred_prob):
    return(np.abs((obs_binary * pred_prob).mean() - ((1 - obs_binary) * pred_prob).mean()))


def replace_many(string, to_replace, replace_with):
    '''
    to_replace is a list
    replace_with is a string
    '''
    for r in to_replace:
        string = string.replace(r, replace_with)
    return(string)

def CI95(est, stderr):
    return([est - 1.96 * stderr, est + 1.96 * stderr])


def ENoRMSE():
    pass


def bootstrap(df, cols=[], B=1000, seed=12345):

    if len(cols) == 0:
        cols = [col in col in df.columns]

    for col in cols:
        for i in range(B):
            df['boot{}_'.format(i)+col] = np.array(df[col].sample(frac=1., replace=True, random_state=seed)).reshape(-1, 1)
    return(df)

# def bootstrap_diff95CI(df, cols, funcs, B=1000, seed=12345):
    
#     df_boot = df.copy()
#     df_boot = bootstrap(df_boot, cols=cols, B=B, seed=seed)
#     measure1 = df_boot[[col for col in df_boot.columns if 'boot' in col and cols[0] in col]].apply(np.std, axis=0)
#     measure2 = df_boot[[col for col in df_boot.columns if 'boot' in col and cols[1] in col]].apply(lambda df_col: np.sqrt(df_col).mean(), axis=0)
#     print(measure1)
#     exit()

def bootstrap_diff95CI(df, cols, funcs, B=1000, seed=12345):
    
    np.random.seed(seed)
    seeds = np.random.randint(B ** 2, size=B)

    diffs = []
    for b in range(B):
        boot1 = (np.array(funcs[0](df[cols[0]].sample(frac=1., replace=True, random_state=seeds[b]))).reshape(-1, 1))
        boot2 = (np.array(funcs[1](df[cols[1]].sample(frac=1., replace=True, random_state=seeds[b]))).reshape(-1, 1))
        diffs += [boot1 - boot2]
    empirical_diffs = pd.DataFrame(np.array(diffs).reshape(-1, 1))
    empirical_diffs.columns = ['diff']
    CI95 = [empirical_diffs.quantile(.025)[0], empirical_diffs.quantile(.975)[0]]
    return(empirical_diffs, CI95)


def var_diff_CI95(df, cols, funcs, estimator, pred_method, B, params, suffix=''):
    scenario_diff = []

    for r2 in [.1, 1.]:
        for r4 in [.1, 1.]:
            for n, p in zip([1000., 5000.], [40., 200.]):

                df_ = df[cols].iloc[np.where((df['r2']==r2) & (df['r4']==r4) & (df['p']==p))[0], :]
                scen = bootstrap_diff95CI(df_, cols, funcs=funcs, B=B)[1]
                indexs = [.1, r2, .1, r4, n, p]
                scenario_diff += [indexs + scen]

    scenario_diff = pd.DataFrame(scenario_diff)
    scenario_diff.columns = params + [suffix+'LL', suffix+'UL']
    scenario_diff['est'] = estimator
    scenario_diff['pred'] = pred_method
    scenario_diff = scenario_diff.set_index(params)
    return(scenario_diff)


def many_95CI(df, many_col_pairs, funcs, B, all_ests, all_preds, params, suffix=''):
    temp = []

    for cols in many_col_pairs:
        temp_methods = ' '.join(cols)
        for est in all_ests:
            if est[1:] in temp_methods:
                for pred in all_preds:
                    if pred[1:] in temp_methods:
                        break
                break
        # if 'nDR' in temp_methods:
        #     suffix = 'exclude'
        temp += [var_diff_CI95(df=df, cols=cols, funcs=funcs, estimator=est, pred_method=pred, B=B, params=params, suffix=suffix)]

    out_df = pd.concat(temp, axis=0)
    return(out_df)


def icd2tfidf(longformatDF, unique_keys=[], icd_col_label=''):
    pass
    # out = [(idx, lst) for idx ]

def up_down_cap(prob_testor):

    np_prob = prob_testor.to("cpu").data.numpy().flatten()

    if np.sum((np_prob > 1.)) > 0 or np.sum((np_prob < 0.)) > 0:
        raise ValueError('the input should be between zero and one.')

    print(prob_testor[prob_testor > .999])
    print(prob_testor[prob_testor < .0001])
    prob_testor[prob_testor > .999] = prob_testor[prob_testor > .999] - .0001
    prob_testor[prob_testor < .0001] = prob_testor[prob_testor < .0001] + .0001
    print(prob_testor[prob_testor < .0001])
    print(prob_testor[prob_testor > .999])

    return(prob_testor)












# import torchsample

def plot(param_dict, layer_names, num_plots):
	"""
	layer_names is the title of weights in the state_dict() for different layers.
	num_plots is the number of histograms for each layer
	"""

	params = param_dict

	#since params is a dictionary of tensors, to get the size of each tensor saved in it, we'll use .size()
	for layer in layer_names:
		if params[layer].size(0) < num_plots:
			raise AssertionError #"Number of plots for a layer should be less than or equal to the size of that layer."
	
	
	fig, multi_plots = plt.subplots(nrows=len(layer_names), ncols=num_plots, sharex=True)

	for i in range(len(layer_names)):
		for j in range(num_plots):
			multi_plots[i, j].hist(params[layer_names[i]][j, :])


	if not os.path.exists("saved_plots"):
		os.makedirs("saved_plots")
	plt.savefig('./saved_plots/mlp_mnist.png')
	plt.show()
	

def get_batch(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = x.shape[0]
    assert N == y.shape[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :]
        batch_y = y[i:i+batch_size]
        yield (batch_x, batch_y)


def plot_learning_curve(train_loss, valid_loss):
	
	e = train_loss.shape[0]
	plt.plot(range(e), train_loss)
	plt.plot(range(e), valid_loss)
	plt.show()

	

def binary_accuracy(y, t, threshold = .5):
	"""y and t are tensors"""
	y_cat = 0 + (y >= threshold)
	#a11 = torch.sum(y_cat * t); a12 = torch.sum(y_cat * (1 - t))
	#a21 = torch.sum((1 - y_cat) * t); a22 = torch.sum((1 - y_cat) * (1 - t))
	a11 = torch.dot(y_cat, t); a12 = torch.dot(y_cat, (1 - t))
	a21 = torch.dot((1 - y_cat), t); a22 = torch.dot((1 - y_cat), (1 - t))
	print("Confusion matrix (predicted vs. observed):")
	confuse = torch.Tensor([[a11, a12], [a21, a22]])
	print(confuse)
	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
	print("Accuracy (%):", np.round(100*(a11 + a22)/torch.sum(confuse), 1))

#def accuracy(y, t, threshold = .5):
#	"""y and t are tensors"""
#	y = y.data.numpy()
#	t = t.data.numpy()
#	y_cat = 0. + (y >= threshold)
#	a11 = np.dot(y_cat.T, t);
#	a12 = np.dot(y_cat.T, (1. - t))
#	a21 = np.dot((1. - y_cat).T, t)
#	a22 = np.dot((1. - y_cat).T, (1. - t))
#	print("Confusion matrix (predicted vs. observed):")
#	#confuse = np.array([[a11, a12], [a21, a22]])
#	confuse = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))
#	print(confuse)
#	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
#	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
#	print("Accuracy (%):", np.round(100*(a11 + a22)/np.sum(confuse), 1))
#	return("------------------------------------------")

def ROC_AUC(predprob, observed):
	fpr, tpr, _ = roc_curve(observed, predprob)
	auc = auc(fpr, tpr)
	return(fpr, tpr, auc)


###############################################################################
###############################################################################
# Simulating data
###############################################################################
###############################################################################


def AR_corr(dim, rho):

	d = np.identity(dim)
	for i in range(dim-1):
		for j in range(i+1, dim):
			d[i, j] = rho**(np.abs(i-j))
			d[j, i] = d[i, j]
	return(d)


def constant_corr(dim, rho):

	all_equal = rho*np.ones((dim, dim))
	d = all_equal - (rho-1)*np.identity(dim)
	
	return(d)



def simulate_data(n, p1, p, error_std, rho, mean, sd, r, type, pr, corr="AR(1)"):
	
	if p1 > p:
		raise ValueError("Number of true signals should be less than or equal to number of inputs: p >= p1")

	if rho == 0.:
		# uncorrelated inputs
		x = mean + sd * np.random.normal(0., 1., size=(n, p))
		# x = mean + sd * np.random.uniform(0., 1., size=(n, p))
	else:
		# correlated inputs 
		means = np.repeat(mean, p)
		if corr.lower() == "constant":
			cov = (sd ** 2) * constant_corr(p, rho=rho)
		elif corr.upper() == "AR(1)":
			cov = (sd ** 2) * AR_corr(p, rho = rho)
		cov_sqrt = np.linalg.cholesky(cov)
		x = means + np.dot(np.random.normal(0., 1., size = (n, p)), cov_sqrt)

	if r[0].lower() == "uniform":
		true_w = np.random.uniform(r[1], r[2], size=(p1, 1))
		negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
		negate[np.where(negate==0.), :] = -1
		true_w = true_w * negate
	elif r[0].lower() == "normal":
		true_w = np.random.nromal(r[1], r[2], size=(p1, 1))
	
	true_index = np.random.choice(np.arange(p), size = p1, replace=False)
	true_index = np.sort(true_index)
	# true_index = np.arange(p1)
	xbeta = np.dot(x[:, true_index], true_w)

	if type.lower() == "regression":
		y = xbeta + error_std * np.random.normal(0., 1., size=(n, 1))
	elif type.lower() == "classification":
		pr = 1. / (1. + np.exp(-xbeta))
		y = np.random.binomial(1, pr, size=x.shape[0])
	
	A = np.random.binomial(1, pr, size=(n, 1))
	return(A, x, y, true_index)


def simulate_x(n, p, rho, mean, sd, corr="AR(1)", dist='normal', seed=123451234):

    np.random.seed(seed)
    # x = np.random.uniform(-.1, .1, size=(n, p))
    # for j in range(int(p/2)):
    #     x[:, j] = (x[:, j] > 0) + 0.

    if rho == 0.:
        # uncorrelated inputs
        if dist == 'normal':
            x = mean + sd * np.random.normal(0., 1., size=(n, p))            
        elif dist == 'uniform':
            x = mean + sd * np.random.uniform(0., 1., size=(n, p))
        # for j in range(int(p/2)):
        #     x[:, j] = (x[:, j] > 0) + 0.
    else:
        # correlated inputs 
        means = np.repeat(mean, p)
        if corr.lower() == "constant":
        	cov = (sd ** 2) * constant_corr(p, rho=rho)
        elif corr.upper() == "AR(1)":
        	cov = (sd ** 2) * AR_corr(p, rho = rho)
        cov_sqrt = np.linalg.cholesky(cov)
        x = means + np.dot(np.random.normal(0., 1., size = (n, p)), cov_sqrt)

    return(x)



def func1(x1, x2):
	return(np.exp((x1 + x2)/ 2.))

def func2(x1, x2):
	return(x1 / (1 + np.exp(x2)))

def func3(x1, x2):
	return((x1 * x2 / 10 + 2)**3)

def func4(x1, x2):
	return((x1 + x2 + 3)**2)

def func5(x1, x2):

	g = 0. + -2 * (x1 < -1) - (-1 < x1)*(x1 < 0) + (x1 < 2)*(0 < x1)+ 3 * (x1 > 2)
	h = 0. + -3 * (x2 < 0) - 2 * (x2 < 1)*(0 < x2) + 5 * (x2 > 1)
	return(g * h)

def func6(x1, x2):

	g = 0. + (x1 > 0)
	h = 0. + (x2 > 1)
	return(g * h)
	

def nonlinear(x, nonlinearity_portion=.2, seed_funcs=1234321):

	np.random.seed(seed_funcs)

	nlf = {1: func1, 2: func2, 3: func3, 4: func4, 5: func5, 6: func6} # Non-linear functions

	n, p = x.shape

	p_nl = int(nonlinearity_portion * p)
	if p_nl == 0.:
		return(x, 'no non-linearity: latents', 'no non-linearity: r', 'no non-linearity: arg1', 'no non-linearity: arg2')
	else:
		r = np.random.choice(len(nlf), p_nl, replace=True) + 1
		arg1 = np.random.choice(p, size=p_nl, replace=True)
		arg2 = np.random.choice(p, size=p_nl, replace=True)
		latents = [nlf[k](x[:, j], x[:, jj]).reshape(-1, 1)  for k, j, jj in zip(r, arg1, arg2)]

		latents = np.concatenate(latents, axis=1)
		x_nonlinear = np.hstack((x[:, p_nl:], latents))

		return(x_nonlinear, latents, r, arg1, arg2)



def reverse_nonlinear(x, nonlin_funcs, set1, set2):

	nlf = {1: func1, 2: func2, 3: func3, 4: func4, 5: func5, 6: func6} # Non-linear functions
	p_nl = len(nonlin_funcs)
	n, p = x.shape

	r, arg1, arg2 = np.array(nonlin_funcs), np.array(set1), np.array(set2)

	latents = [nlf[k](x[:, j], x[:, jj]).reshape(-1, 1)  for k, j, jj in zip(r, arg1, arg2)]

	latents = np.concatenate(latents, axis=1)
	x_nonlinear = np.hstack((x[:, p_nl:], latents))

	return(x_nonlinear)


def simulate_params(p1, r, seed_beta=12345):
    np.random.seed(seed_beta)

    if r[0].lower() == "uniform":
        true_w = np.random.uniform(r[1], r[2], size=(p1, 1))
        negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
        negate[np.where(negate==0.), :] = -1
        true_w = true_w * negate
        
    return(true_w.reshape(-1, 1))

###############################################################################
###############################################################################
# FDR, power and FNP 
###############################################################################
###############################################################################


def FDR(S, TrueIndex):
	"""S is a set of integers (indexes of rejected hypotheses)
	TrueIndex is set of indeces of true signals.
	p is the number of hypotheses.
	"""
	FalseRej = [ix for ix in S if ix not in TrueIndex]
	Rej = float(S.shape[0])#number of rejections 
		
	return(np.round(1. * len(FalseRej)/max(1., Rej), 2))

def power(S, TrueIndex):
	"""S is a set of integers (indexes of rejected hypotheses)
	TrueIndex is set of indeces of true signals.
	"""
	TrueRej = [ix for ix in S if ix in TrueIndex]
	TrueIndex_size = float(TrueIndex.shape[0])
	return(np.round(1. * len(TrueRej)/max(1., TrueIndex_size), 2))

def FNP(S, TrueIndex, p):
	"""S is a set of integers (indexes of rejected hypotheses)
	TrueIndex is set of indeces of true signals.
	p is the number of hypotheses.
	""" 
	NotRej = [ix for ix in range(p) if ix not in S]
	FalseNotRej = [ix for ix in TrueIndex if ix in NotRej]
	return(np.round(1. * len(FalseNotRej) / max(1, len(NotRej)), 2))


###############################################################################
###############################################################################
# Optimization for knockoff method
###############################################################################
###############################################################################

def is_pos_semi_def(x):
		return(np.all(np.linalg.eigvals(x) >= 0.))

def bisection(Sigma, s_hat):

	gamma = 1.
	matrix = 2 * Sigma - np.diag(gamma * s_hat)

	if is_pos_semi_def(matrix):
		return(s_hat)
	
	tol = 1.;	low = 0.;	high = 1.

	while tol > .01:
		matrix = 2 * Sigma - np.diag(gamma * s_hat)
		if is_pos_semi_def(matrix):
			low = gamma
		else:
			high = gamma
		old_gamma = gamma
		gamma = (low + high)/2
		tol = np.abs(old_gamma - gamma)
	
	return(gamma * s_hat)


def SigmaClusterApprox(Sigma, block_size):
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k	
	
	num_blocks = q if r == 0 else q + 1

	cluster = AgglomerativeClustering(n_clusters=num_blocks, affinity='euclidean', linkage='ward')  
	labels = cluster.fit(Sigma).labels_
	Index = [np.argwhere(labels == j).flatten() for j in range(num_blocks)]

	subSigmas = [np.diag(Sigma[Index[j], Index[j]]) for j in range(num_blocks)]

	# for j in range(num_blocks):
	# 	_, d, Vt = np.linalg.svd(Sigma[:, Index[j]])
	# 	low_dim = np.dot(np.diag(d), Vt)
	# 	print(np.linalg.eigvals(low_dim))
	# 	subSigmas += [low_dim]

	return(subSigmas, Index)

# Sigma = np.round(np.random.uniform(size=(14, 14)), 2)
# Sigma = np.dot(Sigma.T, Sigma)
# # print(Sigma)
# block_size = 5
# # print(SigmaClusterApprox(Sigma, block_size))

# print(asdp(Sigma, block_size))

def SigmaBlocksApprox(Sigma, block_size):
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k
	
	subSigmas = [Sigma[(j*k):(j*k+k), (j*k):(j*k+k)] for j in range(q)]
	if r != 0:
		subSigmas += [Sigma[-r:, -r:]]

	return(subSigmas)


def SigmaEigenApprox(Sigma, block_size):
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k
	
	eigenvals = np.linalg.eigvals(Sigma)
	eigenvals[eigenvals < 0.] = 0.
	subSigmas = [np.diag(eigenvals[(j*k):(j*k+k)]) for j in range(q)]
	if r != 0:
		subSigmas += [np.diag(eigenvals[-r:])]
	return(subSigmas)


def sdp(Sigma):
	p = Sigma.shape[0]
	identity_p = np.identity(p)
	zero_one = np.repeat(0., p**3)
	zero_one2 = zero_one.copy()
	indexes = np.arange(p)*(1 + p + p**2)# np.arange(p)*p+ np.arange(p)*(p**2) + np.arange(p)
	zero_one[indexes] = 1.
	block_identity = zero_one.reshape(p*p, p)
	zero_one2[indexes] = -1.
	mblock_identity = zero_one2.reshape(p*p, p)
	
	c = matrix(np.repeat(-1., p))
	G = [matrix(block_identity)] + [matrix(mblock_identity)] + [matrix(block_identity)]
	h = [matrix(2.*Sigma)] + [matrix(np.zeros((p, p)))] + [matrix(identity_p)]
	solvers.options['show_progress'] = False
	sol = solvers.sdp(c, Gs=G, hs=h)['x']
	sol = np.array(sol).flatten()
	sol[sol > 1.] = 1.
	sol[sol < 0.] = 0.
	# print(os.getpid(), "\n")
	
	return(sol)


def asdp(Sigma, block_size, approx_method):
	
	approx_method = approx_method.lower()
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k

	if approx_method == "selfblocks":
		subSigmas = SigmaBlocksApprox(Sigma, block_size)
	elif approx_method == "cluster":
		subSigmas, Index = SigmaClusterApprox(Sigma, block_size)

	num_blocks = q if r == 0 else q + 1
	
	#########parallel computing###########
	# pool = multiprocessing.Pool()
	# subSolutions_ = [pool.apply_async(sdp, (sub_mat, )) for sub_mat in subSigmas]
	# pool.close()
	# subSolutions = [sols.get() for sols in subSolutions_]
	subSolutions = [sdp(sub_mat) for sub_mat in subSigmas]
	
	if approx_method == "cluster":
		ordered_subSolutions = np.repeat(0., p)
		for j in range(num_blocks):
			ordered_subSolutions[Index[j]] = subSolutions[j]

		subSolutions = [ordered_subSolutions]

	s_hat = np.concatenate(subSolutions)

	s = bisection(Sigma, s_hat)
	s[s > 1.] = 1.
	s[s < 0.] = 0.			
	
	return(s)
	
