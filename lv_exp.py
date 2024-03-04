import random 
from distributions import *
from sympy.parsing.sympy_parser import T
from utils import *
import torch
import torch.distributions as tdist
from neuralStein import *
from LangevinSampler import *
import pandas as pd
import seaborn as sns
from neural_CV import *
import numpy as np
from scipy.integrate import odeint
from scipy.stats import lognorm, norm
import torchdiffeq

# Set the random seed for reproducible results
torch.manual_seed(23) 

# Set the aesthetic style of the plots
sns.set(style="whitegrid", palette="pastel")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARAM_TO_ESTIMATE = 1 # This gives the index of the parameter to estimate : 0 = x_1, 1 = x_2, 2 = x_3, 3 = x_4
N = 1024
EPOCHS = 1000

means = torch.tensor([0, -3, -3, 0, np.log(10), np.log(10), -1, -1]).to(device).float()
variances = torch.tensor([0.5**2, 0.5**2, 0.5**2, 0.5**2, 1**2, 1**2, 1**2, 1**2]).to(device).float()
covs = torch.diag(variances)
data = pd.read_csv('/content/NeuralSteinExpectations/hudson-bay-lynx-hare.csv',header=2)
data = data.to_numpy()
t = data[:, 0]
y_obs = data[:, 1:]

TRUE_MEAN = means[PARAM_TO_ESTIMATE].float()
TRUE_COV = covs[PARAM_TO_ESTIMATE,PARAM_TO_ESTIMATE].float()

class LVdist :
    def __init__(self, means, covs, t, y_obs):
        self.prior = NormalDistribution(means[PARAM_TO_ESTIMATE], covs[PARAM_TO_ESTIMATE,PARAM_TO_ESTIMATE])
        self.t = torch.tensor(t, requires_grad=True, device=device)
        self.y_obs = torch.tensor(y_obs, requires_grad=True, device=device)
        self.dim = 1
        self.x = means.unsqueeze(1)

    def log_prob(self, x):
        # Reparameterize
        if self.x.size(1) != x.size(0):
                self.x = self.x.repeat(1,x.size(0))
        

        self.x[PARAM_TO_ESTIMATE,:] = x[:,0]
        x_tilde = torch.exp(self.x[:4])
        initial_pop_tilde = torch.exp(self.x[4:6])
        sigma_tilde = torch.exp(self.x[6:8])

        # Solve ODE
        u0 = initial_pop_tilde
        ode_system = lambda t, u: self.lotka_volterra_system(t, u, x_tilde)
        u = torchdiffeq.odeint(ode_system, u0, self.t)

        # Log-likelihood for the observed data
        log_likelihood = 0
        eps = 1e-9 # To avoid log(0)
        # Select the solution corresponding to the i-th parameter set
        u_i = u[:, :, PARAM_TO_ESTIMATE]
        # Calculate log-likelihood for this parameter set
        log_likelihood = -0.5 * torch.sum((torch.log(u_i+eps) - torch.log(self.y_obs+eps))**2 / sigma_tilde[:,PARAM_TO_ESTIMATE]**2)

        # Log prior using NormalDistribution
        log_prior = self.prior.log_prob(x)

        # Total log probability
        log_prob = log_likelihood + log_prior
        return log_prob  # Return as a tensor for optimization

    def lotka_volterra_system(self, t, u, x_tilde):
        du1_dt = x_tilde[0] * u[0] - x_tilde[1] * u[0] * u[1]
        du2_dt = x_tilde[2] * u[0] * u[1] - x_tilde[3] * u[1]
        return torch.stack([du1_dt, du2_dt])
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, self.dim) * (sample_range[1] - sample_range[0]) + sample_range[0]


dist = LVdist(means, covs, t, y_obs)

def integrand(x):
    return x

dim = 1 
post_dist = tdist.Normal(TRUE_MEAN,TRUE_COV)
post_samples = post_dist.sample((max(2*N,3000),1))
sample_range = (TRUE_MEAN.cpu().detach().numpy()-5,TRUE_MEAN.cpu().detach().numpy()+5)

# the training mesh is uniformly sampled from hypercube [-10, 10]^dim, n_samples total
stein_est = evaluate_stein_expectation(dist, 
                           dim,
                           sample_range= sample_range, 
                           n_samples = N, 
                           h =integrand,
                           epochs=EPOCHS,
                           loss_type = "diff")

stein_est_given_samples = evaluate_stein_expectation(dist, 
                           dim,
                           sample_range= sample_range, 
                           n_samples = N, 
                           h =integrand,
                           epochs=EPOCHS,
                           loss_type = "diff",
                           given_sample = post_samples[:N].to(device))

# compare to NCV
ncv_est_given_samples = evaluate_ncv_expectation(dist, 
                           dim,
                           sample_range= sample_range, 
                           n_samples = N, 
                           h =integrand,
                           epochs=EPOCHS,
                           given_sample = post_samples[:N].to(device))  

ncv_est = evaluate_ncv_expectation(dist, 
                           dim,
                           sample_range= sample_range, 
                           n_samples = N, 
                           h =integrand,
                           epochs=EPOCHS)  

f_vals_true_samples = integrand(post_samples).detach().cpu().numpy()
post_samples_est = f_vals_true_samples.mean()


print(f'Analytic true moment: {TRUE_MEAN}, MC Sampled est: {post_samples_est} \n NCV estimates : \n\t- true samples: {ncv_est_given_samples}, \n\t- using off-samples: {ncv_est} \n Stein estimates : \n\t- true samples: {stein_est_given_samples}, \n\t- using off-samples: {stein_est}')
