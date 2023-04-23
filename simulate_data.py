
import numpy as np 

from utils import simulate_x, nonlinear, simulate_params

from sklearn.preprocessing import StandardScaler
import time
import random
import matplotlib.pyplot as plt



'''

Simulate 4 types of observed covariates: X_c are a set of covariates that influence both the treatment and outcome, also called confouders. X_iv are a set of covariates only influence treatment and not outcome, also called nstrumental variables. X_y are a set of covariates only influence the outcome and not treatment. X_irr are a set of covariates influencing neither of treatment and outcome.

Accordingly, we define the number of covariates (dimension of columns) as p_c, p_iv, p_y, p_n

We fix the correlation between the covarites in each set to follow the same pattern with the base correlation rho. The correlation pattern is so that corr(X_j, X_k) = rho^(j-k). This is known as auto-regressive, AR(1).

We let the range of signals sizes be between r1 and r2.

We let the treatment be a binary treatment with probability of treated to be pr.

Let's agument some nonlinear functions of the columns in each covariate matrix:

f1(x1, x2) = exp(x1*x2/2)

f2(x1, x2) = x1/(1+exp(x2))

f3(x1, x2) = (x1*x2/10+2)^3

f4(x1, x2) = (x1+x2+3)^2

f5(x1, x2) = g(x1) * h(x2)

where

g(x) = -2 I(x < -1) - I(-1 < x < 0) + I(0 < x < 2)+ 3 I(x > 2)

h(x) = -5 I(x < 0) - 2 I(0 < x < 1) + 3 * I(x > 1)

and

f6(x1, x2) = g(x1) * h(x2)

where

g(x) = I(x > 0)

h(x) = I(x > 1)

We let "nonlinearity_portion" proportion of covariates in each matrix to be replaced by randomly selected nonlinearities. These nonlinearities are bivariate. We randomly select two of the covariates each time and implement the nonlinearities.

'''
from scipy.stats import genextreme

def SimulateNoIntraction(True_TE, n, p_c, p_iv, p_y, p_n, rho=.5, corr="AR(1)", nonlinearity_portion=.20, r1=.1, r2=1., r3=.1, r4=1., 
	sigma=1., seed=12345, seed_beta=12345, seed_funcs=12345, plot=False, 
	dist='normal', link_func='sigmoid'):
	
	random.seed(seed)

	r = ["uniform", r1, r2]
	r_iv = ["uniform", r3, r4]

	X_irr = simulate_x(n=n, p=p_n, rho=rho, mean=0., sd=1, corr="AR(1)", seed=int(str(seed)+'1'), dist=dist)

	A = np.zeros((n, 1))
	J = 0
	
	param_iv = simulate_params(p1=p_iv, r=r_iv, seed_beta=seed_beta)

	while (A.mean() < .25 or A.mean() > .75) and J < 10:
		# param_iv = simulate_params(p1=p_iv, r=r_iv, seed_beta=seed_beta + J)
		X_c = simulate_x(n=n, p=p_c, rho=rho, mean=0., sd=1, corr="AR(1)", seed=int(str(seed + J)+'2'), dist=dist)
		X_iv = simulate_x(n=n, p=p_iv, rho=rho, mean=0., sd=1, corr="AR(1)", seed=int(str(seed + J)+'3'), dist=dist)

		X_c_latent, latent_c, funcs_c, arg1_c, arg2_c = nonlinear(X_c, nonlinearity_portion=nonlinearity_portion, seed_funcs=seed_funcs) # nonlinearity_portion is the proportion of columns that are replaced by nonlinear functions.
		X_iv_latent, latent_iv, funcs_iv, arg1_iv, arg2_iv = nonlinear(X_iv, nonlinearity_portion=nonlinearity_portion, seed_funcs=seed_funcs)

		xbeta_iv = np.dot(np.hstack((X_iv, X_c)), np.vstack((param_iv, param_iv)))

		if link_func == 'sigmoid':
			pr = 1./(1. + np.exp(-xbeta_iv))
		elif 'genextreme' in link_func:
			pr = genextreme.cdf(xbeta_iv, link_func[1])
			plt.hist(pr, bins=20, density=True)
			plt.show()
		A = np.random.binomial(1, pr, size=(n, 1)) 
		J += 1

	if plot:
		xbeta_conf_only = np.dot(X_c_latent, param_iv)
		pr_conf_only = 1./(1. + np.exp(-xbeta_conf_only))
		import plotly.graph_objects as go
		from plotly.subplots import make_subplots
		fig = make_subplots(rows=2, cols=2, subplot_titles=['Xbeta, with both conf and IV','pr, with both conf and IV','Xbeta, with only conf','pr, with only conf'])
		trace0 = go.Histogram(x=xbeta_iv.flatten(), nbinsx=20)
		trace1 = go.Histogram(x=pr.flatten(), nbinsx=20)
		trace2 = go.Histogram(x=xbeta_conf_only.flatten(), nbinsx=20)
		trace3 = go.Histogram(x=pr_conf_only.flatten(), nbinsx=20)
		fig.append_trace(trace0, 1, 1)
		fig.append_trace(trace1, 1, 2)
		fig.append_trace(trace2, 2, 1)
		fig.append_trace(trace3, 2, 2)
		fig.show()
		return()
	else:
		X_y = simulate_x(n=n, p=p_y, rho=rho, mean=0., sd=1, corr="AR(1)", seed=int(str(seed)+'4'), dist=dist)

		X_y_latent, latent_y, funcs_y, arg1_y, arg2_y = nonlinear(X_y, nonlinearity_portion=nonlinearity_portion, seed_funcs=seed_funcs)

		xbeta_y = 1 + True_TE * A + np.dot(np.hstack((X_y_latent, X_c_latent)), np.vstack((param_iv, param_iv))).reshape(-1, 1)

		if type(sigma) in [float, int] or sigma in ['const', 'constant']:
			y = xbeta_y + sigma * np.random.normal(size=(n, 1))
		elif sigma in ['uniform']:
			y = xbeta_y + np.random.uniform(size=(n, 1)) * np.random.normal(size=(n, 1))
		elif sigma in ['normal']:
			y = xbeta_y + np.random.normal(size=(n, 1)) * np.random.normal(size=(n, 1))

		index_c = np.arange(p_c)
		index_iv = np.arange(p_c, p_c + p_iv)

		x_ = np.concatenate([X_c, X_iv, X_y, X_irr], axis=1)# Big data with all covarites (excluding th treatment)
		x_in_DGP = {}
		x_in_DGP['X_c_latent'] = X_c_latent
		x_in_DGP['X_iv_latent'] = X_iv_latent
		x_in_DGP['X_y_latent'] = X_y_latent

		sdtz = StandardScaler()
		x = sdtz.fit_transform(x_)

		# return(y, A, x, X_c_latent, X_y_latent, [funcs_c, arg1_c, arg2_c, funcs_iv, arg1_iv, arg2_iv, funcs_y, arg1_y, arg2_y]) # Mean 0 and unit variance
		return(y, A, x, x_in_DGP) # x_in_DGP is the inputs that generate the true y (DGP: data generating process)



# def SimulateWithIntraction(True_TE, n, p_c, p_iv, p_y, p_n, rho=.5, corr="AR(1)", nonlinearity_portion=.10, interaction_portion=.1, r1=.1, r2=1., sigma=1., seed=int(time.time()), seed_funcs=int(time.time())):

# 	np.random.seed(seed)

# 	X_c = simulate_x(n=n, p=p_c, rho=rho, mean=0., sd=.1, corr="AR(1)", dist=dist)
# 	X_iv = simulate_x(n=n, p=p_iv, rho=rho, mean=0., sd=.1, corr="AR(1)", dist=dist)
# 	X_y = simulate_x(n=n, p=p_y, rho=rho, mean=0., sd=.1, corr="AR(1)", dist=dist)
# 	X_irr = simulate_x(n=n, p=p_n, rho=rho, mean=0., sd=.1, corr="AR(1)", dist=dist)

# 	if nonlinearity_portion != 0.:
# 		X_c_latent, funcs_c = nonlinear(X_c, nonlinearity_portion=nonlinearity_portion, seed_funcs=seed_funcs)# nonlinearity_portion is the proportion of columns that are replaced by nonlinear functions.
# 		X_iv_latent, funcs_iv = nonlinear(X_iv, nonlinearity_portion=nonlinearity_portion, seed_funcs=seed_funcs)
# 		X_y_latent, funcs_y = nonlinear(X_y, nonlinearity_portion=nonlinearity_portion, seed_funcs=seed_funcs)
# 	else:
# 		X_c_latent, funcs_c = X_c, 0
# 		X_iv_latent, funcs_iv = X_iv, 0
# 		X_y_latent, funcs_y = X_y, 0

# 	r = ["uniform", r1, r2]

# 	param_iv = simulate_params(p1=p_iv + p_c, r=r)
# 	xbeta_iv = np.dot(np.hstack((X_iv_latent, X_c_latent)), param_iv)
# 	pr = 1./(1. + np.exp(-xbeta_iv))
# 	A = np.random.binomial(1, pr, size=(n, 1))

# 	param_y = simulate_params(p1=p_y + p_c, r=r)

# 	xbeta_y = np.dot(np.hstack((A, np.hstack((X_y_latent, X_c_latent)))), np.vstack((True_TE, param_y))).reshape(-1, 1)

# 	# number of interaction terms
# 	n_interactions = int(nonlinearity_portion*p_c)
# 	x_interactions = A * X_c[:, np.random.choice(p_c, n_interactions, replace=False)]
# 	param_interactions = simulate_params(p1=n_interactions, r=r)

# 	interactions = np.dot(x_interactions, param_interactions).reshape(-1, 1)

# 	y = xbeta_y + interactions + sigma * np.random.normal(size=(n, 1))

# 	index_c = np.arange(p_c)
# 	index_iv = np.arange(p_c, p_c + p_iv)

# 	x_ = np.concatenate([X_c, X_iv, X_y, X_irr], axis=1)# Big data with all covarites (excluding th treatment)

# 	sdtz = StandardScaler()
# 	x = sdtz.fit_transform(x_)
# 	return(y, A, x, [funcs_c, funcs_iv, funcs_y]) # Mean 0 and unit variance


