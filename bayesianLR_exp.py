from distributions import *
from network import MLP
from utils import *
from ToyDistributions import *
import torch.optim as optim
import torch
import torch.distributions as tdist
import random
import math
from neuralStein import *
from LangevinSampler import *
import pandas as pd
from sklearn.metrics import r2_score

DATAROOT = "./Dataset"

def square(x):
    return (x**2).sum(-1)

def identity(x):
    return x.sum(-1)

# use this to compute first and second mean of a 2D vector
def first_comp(x):
    return x[:,0]

def second_comp(x):
    return x[:,1]

#####################################################
# Linear Regression Air Quality Dataset (from UCI)
###################################################

## Air quality dataset
# 7485/5 training points
# 1872 validation points
def get_airquality(batch_size, normalize = True):
    col1=['DATE','TIME','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT','PT08_S2_NMHC',
     'NOX_GT','PT08_S3_NOX','NO2_GT','PT08_S4_NO2','PT08_S5_O3','T','RH','AH']

    df1 = pd.read_excel('./Dataset/AirQualityUCI.xlsx',header=None,skiprows=1, na_filter=True,names=col1)
    #print("Airquality dataframe: " len(df1))
    
    df1 = df1.dropna()
    df1['DATE']=pd.to_datetime(df1.DATE, format='%d-%m-%Y')
    df1['MONTH']= df1['DATE'].dt.month
    df1['HOUR']=df1['TIME'].apply(lambda x: int(str(x).split(':')[0]))
    df1 = df1.drop(columns=['NMHC_GT'])
    df1 = df1.drop(columns=['DATE'])
    df1 = df1.drop(columns=['TIME'])
    df1 = df1.drop(columns=['AH'])
    
    #delete data with 'CO_GT' = -200
    df1 = df1[df1['CO_GT'] != -200]

    df1.to_pickle("./Dataset/AirQualityProcessedDF.pickle")
    col1 = df1.columns.tolist()
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 = (df1 - df1.mean())/(df1.std()) #(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    target = data.pop('CO_GT')
    #data =  data[['PT08_S1_CO']]
    train_size = int((len(data.values) * 0.8) / 5)

    train_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[:train_size, :]).float(), torch.Tensor(target.values[:train_size]).float())
    test_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[train_size:, :]).float(), torch.Tensor(target.values[train_size:]).float())
    
    torch.save(train_ds, "./Dataset/AirQualityTrain.pt")
    torch.save(test_ds, "./Dataset/AirQualityTest.pt")
    
    # We shuffle with a buffer the same size as the dataset.
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle = True
    )
    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle = False
    )
    return trainloader, testloader, train_ds, test_ds


#####################################################
# Linear Regression Synthetic Data
###################################################
VAR = 1.0

TARGET_MU = 0.0
TARGET_VAR = 0.5*VAR

# x data
def gen_data(npoints = 1000, mu_t = TARGET_MU, var_t = TARGET_VAR):
    data = np.random.normal(loc = mu_t, scale = np.sqrt(var_t), size = npoints)
    return data 


# data x is generated and split same way as before, but there is a ground truth global set of weights that generates y (and var_y)
W_TRUE = np.array([-5.0, 10.0]) #y = 10x + -5)

VAR_Y = 1.0 #1.0


def gen_data_regr(npoints = 1000, mu_t_x = TARGET_MU, var_t_x = TARGET_VAR, w_true = W_TRUE, var_y = VAR_Y):
    data_x = gen_data(npoints = npoints, mu_t = mu_t_x, var_t= var_t_x)

    #N, array of y_vals
    data_y = w_true[0] + data_x*w_true[1]

    # add noise to y to get correct predictive variance
    data_y = data_y + np.sqrt(var_y)*np.random.randn(*data_y.shape) 

    return data_x, data_y 

###########################################################################
# Bayesian linear regression implementation 
#########################################################################

PRIOR_PREC = 1e-2 #0.0
LLHD_PREC = 1/VAR_Y #1e10 #1.0

# Exact posterior
#bayesian linear regression, with prior = N(w | 0, prior_prec^(-1) I)
#x_mat a [N x D] design matrix for the training data
def linRegr_posterior(x_mat, data_y, prior_prec = PRIOR_PREC, llhd_prec = LLHD_PREC):
    #compute gram matrix
    K = (x_mat.T @ x_mat)
    #posterior precision matrix
    post_prec = prior_prec*torch.eye(K.shape[0]) + llhd_prec*K 
    post_cov = torch.linalg.inv(post_prec)
    post_mean = llhd_prec * post_cov @ (x_mat.T @ data_y)

    return post_mean, post_cov 

# returns Sn, posterior covariance matrix
def linRegr_posterior_cov(x_mat, prior_prec = PRIOR_PREC, llhd_prec = LLHD_PREC):
    #compute gram matrix
    K = (x_mat.T @ x_mat)
    #posterior precision matrix
    post_prec = prior_prec*torch.eye(K.shape[0]) + llhd_prec*K 
    post_cov = torch.linalg.inv(post_prec)

    return post_cov 

# Exact predictive posterior
#x_input should be design matrix: NxD (ie. 1x 2 for [1 x] inputs)
def linRegr_pred(x_input, post_mean, post_cov, llhd_prec = LLHD_PREC):
    post_pred_mean = post_mean.T @ x_input.T 

    post_pred_var =  x_input @ post_cov @ x_input.T
    post_pred_var = (1/llhd_prec)*torch.eye(post_pred_var.shape[0]) + post_pred_var

    return post_pred_mean, post_pred_var 

def linRegr_pred_var(x_input, post_cov, llhd_prec = LLHD_PREC):
    post_pred_var = x_input @ post_cov @ x_input.T
    post_pred_var = (1/llhd_prec)*torch.eye(post_pred_var.shape[0]) + post_pred_var 

    return post_pred_var 

#data_x is [N,] of data points 
def get_design_mat(data_x):
    if len(data_x.shape) == 1:
        data_x = data_x.reshape(-1, 1)
    x_mat =  torch.cat([torch.ones(data_x.shape[0], 1), data_x], dim=1)
    return x_mat 

def global_linRegr_eval(data_x, data_y, v_data_x, v_data_y, plot_dir = "./plots/"):
    data_x = torch.Tensor(data_x) 
    data_y_torch = torch.Tensor(data_y).reshape(-1, 1) 

    v_data_x_torch = torch.Tensor(v_data_x)
    v_data_y_torch = torch.Tensor(v_data_y)


    x_mat = get_design_mat(data_x)

    post_mean, post_cov = linRegr_posterior(x_mat, data_y_torch)

    pred_means = []
    pred_vars = [] 

    for x in v_data_x:
        x_input = get_design_mat(torch.Tensor([x]).reshape(-1,))
        post_pred_mean, post_pred_var = linRegr_pred(x_input, post_mean, post_cov) 

        pred_means.append(post_pred_mean.item())
        pred_vars.append(post_pred_var.item()) 

    plot_preds(pred_means, pred_vars, v_data_x, v_data_y, title = "Global Ground Truth Predictions P(y|x,D)", plot_dir=plot_dir, save_name = "global_gt_preds")

    log_llhd = eval_log_llhd(v_data_y_torch, pred_means, pred_vars)

    print("Log Likelihood of global model is: {}".format(log_llhd))   
    
    return pred_means, pred_vars

# log p(w | D)
def posteriorLogProb(w, data_x, data_y):
    data_x = torch.Tensor(data_x) 
    data_y_torch = torch.Tensor(data_y).reshape(-1, 1) 

    x_mat = get_design_mat(data_x)

    post_mean, post_cov = linRegr_posterior(x_mat, data_y_torch)

    return torch.distributions.MultivariateNormal(post_mean.reshape(-1), post_cov).log_prob(w)


def posteriorPredLogProb(x_inp, y_inp, data_x, data_y):
    data_x = torch.Tensor(data_x) 
    data_y_torch = torch.Tensor(data_y).reshape(-1, 1) 

    x_mat = get_design_mat(data_x)

    post_mean, post_cov = linRegr_posterior(x_mat, data_y_torch)
    
    x_input = get_design_mat(torch.Tensor([x_inp]).reshape(-1,))
    post_pred_mean, post_pred_var = linRegr_pred(x_input, post_mean, post_cov) 
    
    return torch.distributions.Normal(post_pred_mean.reshape(-1), torch.sqrt(post_pred_var)).log_prob(y_inp)

def plot_preds(pred_means, pred_vars, v_data_x, v_data_y, title,  plot_dir, save_name):

    print(v_data_x)

    plt.figure()
    plt.plot(v_data_x, pred_means, label = "Mean of p(y|x,D) model")
    plt.scatter(v_data_x, v_data_y, label = "True Data")
    plt.fill_between(v_data_x, pred_means - np.sqrt(pred_vars), pred_means + np.sqrt(pred_vars), alpha=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.savefig(plot_dir + save_name +".png")

 
def eval_log_llhd(v_data_y, pred_means, pred_vars):
    covs = torch.Tensor(pred_vars)

    if len(covs.shape) == 1:
        pred_dist = torch.distributions.normal.Normal(loc=torch.Tensor(pred_means), scale=torch.sqrt(torch.Tensor(pred_vars)))
    
    else:
        pred_dist = torch.distributions.MultivariateNormal(pred_means, covs)
    log_llhds = pred_dist.log_prob(v_data_y.reshape(-1))
    log_llhd = log_llhds.sum()
    return log_llhd 


def experiment_bayes_mean(v_data_x, v_data_y):
    v_data_x_torch = torch.Tensor(v_data_x)
    v_data_y_torch = torch.Tensor(v_data_y)

    x_mat = get_design_mat(v_data_x_torch)
    print(f'x_mat shape: {x_mat.shape}')
    post_mean, post_cov = linRegr_posterior(x_mat, v_data_y_torch)

    dim = x_mat.shape[1]
    print(f'Posterior mean shape: {post_mean.shape}')
    print(f'Posterior mean: {post_mean}')

    def distLog(x):
        return posteriorLogProb(x, v_data_x_torch, v_data_y_torch)
    
    dist = CustomDistribution(distLog, dim)
    
    # Estimate E[w[0] | D] and E[w[1] | D]
    h_first = first_comp
    h_second = second_comp

    stein_est_first = evaluate_stein_expectation(dist, dim, (-10,10), 500, h=h_first, epochs=2000)
    stein_est_second = evaluate_stein_expectation(dist, dim, (-10,10), 500, h=h_second, epochs=2000)

    print(f'True posterior mean w[0]: {post_mean[0]}, Stein estimate: {stein_est_first}')
    print(f'True posterior mean w[1]: {post_mean[1]}, Stein estimate: {stein_est_second}')





if __name__ == "__main__":
    trainloader, testloader,  train_ds, test_ds= get_airquality(batch_size = 512, normalize = True)
    data_x, data_y = train_ds.tensors
    v_data_x, v_data_y = gen_data_regr(npoints = 1000, mu_t_x = TARGET_MU, var_t_x = TARGET_VAR, w_true = W_TRUE, var_y = VAR_Y)
    experiment_bayes_mean(v_data_x, v_data_y)