from distributions import *
from network import MLP
from utils import *
from ToyDistributions import *
import torch.optim as optim
import torch
import torch.distributions as tdist
import random
import math
from OtherMethods import HamiltonianMCMC 
from neuralStein import *
from LangevinSampler import *
import pandas as pd

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
    
    #delete data with 'CO_GT' = -200
    df1 = df1[df1['CO_GT'] != -200]

    df1.to_pickle("./Dataset/AirQualityProcessedDF.pickle")
    col1 = df1.columns.tolist()
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 = (df1 - df1.mean())/(df1.std()) #(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    target = data.pop('CO_GT')

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

PRIOR_PREC = 1e-4 #0.0
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
    data_x = data_x.reshape(-1)

    x_mat = torch.stack([torch.ones_like(data_x), data_x], axis=1)

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

def experiment_bayes_mean(data_x, data_y, v_data_x, v_data_y):
    data_x = torch.Tensor(data_x) 
    data_y_torch = torch.Tensor(data_y).reshape(-1, 1) 

    v_data_x_torch = torch.Tensor(v_data_x)
    v_data_y_torch = torch.Tensor(v_data_y)

    x_mat = get_design_mat(data_x)
    
    post_mean, post_cov = linRegr_posterior(x_mat, data_y_torch)

    dim = 2

    def distLog(x):
        return posteriorLogProb(x, data_x, data_y)
    dist = CustomDistribution(distLog, dim)
    # estimate E[w[0] | D]
    h = first_comp

    stein_est = evaluate_stein_expectation(dist, dim,(-10,10), dim*300, h=h, epochs=1000*(int(dim/10)+1))
        
    langevin_est = eval_Langevin(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)
    hmc_est = eval_HMC(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)

    # since the moment sums over each dimension, the true moment is the sum of the moments for each dimension
    true_moment = post_mean[0]

    print(f'True posterior mean w[0]: {true_moment}, Stein estimate: {stein_est}, Langevin estimate: {langevin_est}, HMC estimate: {hmc_est}')

    # estimate E[w[1] | D]
    h = second_comp

    stein_est = evaluate_stein_expectation(dist, dim,(-10,10), dim*300, h=h, epochs=1000*(int(dim/10)+1))
        
    langevin_est = eval_Langevin(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)
    hmc_est = eval_HMC(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)

    # since the moment sums over each dimension, the true moment is the sum of the moments for each dimension
    true_moment = post_mean[1]

    print(f'True posterior mean w[1]: {true_moment}, Stein estimate: {stein_est}, Langevin estimate: {langevin_est}, HMC estimate: {hmc_est}')


def experiment_bayes_pred(data_x, data_y, v_data_x, v_data_y, verbose= True):
    data_x = torch.Tensor(data_x) 
    data_y = torch.Tensor(data_y).reshape(-1, 1) 
    
    num_epochs = 1000
    minibatch_size = 100

    h = identity

    # Initialize distribution and MLP network
    # 1 input = x, 1 input is y (distribution over y, but x is "amortization" parameter)
    net = MLP(n_dims=2, n_out=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


    # train amortized stein network
    for epoch in range(num_epochs):
        # sample a minibatch of data_x
        minibatch_idx = random.sample(range(len(data_x)), minibatch_size)

        # get the minibatch of data_x
        minibatch_x = data_x[minibatch_idx]
        # get the minibatch of data_y
        minibatch_y = data_y[minibatch_idx]
        y_samples = minibatch_y #torch.linspace(torch.min(minibatch_y[:,0]), torch.max(minibatch_y[:,0]))
        y_samples.requires_grad = True
        
        # get the design matrix for the minibatch
        x_mat = get_design_mat(minibatch_x)
        
        def logprob(y):
            return posteriorPredLogProb(minibatch_x, y, data_x, data_y)
        
        def net_x(y):
            # connect the y samples with the minibatch of x
            # stack them along dimension -1
            xy = torch.stack([minibatch_x, minibatch_y], axis=-1)

            return net(xy)

        optimizer.zero_grad()

        stein_val = stein_g(y_samples, net_x, logprob)

        grad_s = get_grad(stein_val.sum(), y_samples)
        grad_h = get_grad(h(y_samples).sum(), y_samples)

        loss = torch.sum((grad_s - grad_h)**2)
        loss.backward()
        optimizer.step()
        if verbose:
            if epoch % 100 == 0:  
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

    def net_vx(y):
        # connect the y samples with the minibatch of x
        # stack them along dimension -1
        xy = torch.stack([v_data_x, y], axis=-1)

        return net(y)
    # calculate 
    mean_Stein_preds = h(v_data_y) - stein_g(v_data_y, net_vx, posteriorPredLogProb)

    plt.plot(mean_Stein_preds)



if __name__ == "__main__":
    data_x, data_y = get_airquality(batch_size = 512, normalize = True)
    v_data_x, v_data_y = gen_data_regr(npoints = 1000, mu_t_x = TARGET_MU, var_t_x = TARGET_VAR, w_true = W_TRUE, var_y = VAR_Y)

    experiment_bayes_mean(data_x, data_y, v_data_x, v_data_y)