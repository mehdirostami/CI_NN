
import numpy as np 


# print(0.919822*np.sqrt(0.798336-.5)       ,0.938127*np.sqrt(0.924143-.5))
# print(0.919822*np.sqrt(0.798336-.5)*np.sqrt(1-0.798336)       ,0.938127*np.sqrt(0.924143-.5)*np.sqrt(1-0.924143))


# print('-------------------')

# print(0.936926*np.sqrt(0.943372-.5)       ,0.92198*np.sqrt(0.940967-.5))
# print(0.936926*np.sqrt(0.943372-.5)*np.sqrt(1-0.943372)       ,0.92198*np.sqrt(0.940967-.5)*np.sqrt(1-0.940967))


# print('-------------------')

# print(np.sqrt(0.748029-.5)*np.sqrt(1-0.748029)       ,np.sqrt(0.826538-.5)*np.sqrt(1-0.826538))

# print('-------------------')

# print(np.sqrt(0.811814-.5)*np.sqrt(1-0.811814)       ,np.sqrt(0.925331-.5)*np.sqrt(1-0.925331))




# exit()
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

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.optimize import nnls

import suplearn
import pylab
import matplotlib.pyplot as plt
import time
import utils
import training
import os

import Rsuperlearner

import warnings
warnings.filterwarnings('ignore')
# import packages needed to run r code in python:



def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("niter", help="Iteration number.", type=int)
    parser.add_argument("fold", help="Number of folds for CV.", type=int)
    parser.add_argument("n", help="Sample size.", type=int)
    parser.add_argument("p", help="covariate size", type=int)
    parser.add_argument("lb", help="lower bound of adjusting factors' effects", type=float)
    parser.add_argument("ub", help="upper bound of adjusting factors' effects", type=float)
    args = parser.parse_args()

    niter = args.niter
    split_num = args.fold
    n = args.n
    p = args.p
    lb = args.lb
    ub = args.ub

    if n == 1000:
        data_dims = [(1000, 10, 10, 10, 10)] 
    else:
        data_dims = [(5000, 50, 50, 50, 50)] 

    r = [(lb, ub)]

    
    path = "/home/mr/PhD/Causality in AI/Sims2020"

    print('SUPL, Iteration: ', niter)
    try:
        for r1, r2 in r:

            scenario_name = "featureImp_dim{}_{}_r_{}_{}".format(str(n), str(p), str(r1), str(r2))

            temp_data = pd.read_csv("{}/{}/KfoldQ_ps_nns{}.csv".format(path, scenario_name, niter))

            temp_data, other_ests, supl_run_time, coefs_g, coefs_Q = Rsuperlearner.estimate_supl(temp_data, n, p, fold=split_num, scenario=scenario_name) 
                            
            temp_data.to_csv("{}/{}/KfoldQ_ps_nns{}.csv".format(path, scenario_name, niter), index=False)
    except:
        pass



if __name__ == "__main__":
    Main()