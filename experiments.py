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


def square(x):
            return (x**2).sum(-1)

def identity(x):
    return x.sum(-1)

def exp_compare_dim_Gaussian():
    
    h = square
 
    dims = [1, 2, 10, 50, 100]
    MEAN  = 3.0
    STD = 5.0

    true_moments = [(MEAN**2 + STD**2)*dim for dim in dims]

    stein_ests = []
    langevin_ests = []
    hmc_ests = []   

    stein_errors = []
    langevin_errors = []
    hmc_errors = []

    # set up distribution, a Multivariate Gaussian
    for dim in dims:
        dist = MultivariateNormalDistribution(mean = MEAN+torch.zeros(dim),
                                              covariance=(STD**2)*torch.eye(dim)
                                              ) 
        #torch.distributions.MultivariateNormal(MEAN+torch.zeros(dim), (STD**2)*torch.eye(dim))

        # Evaluate Stein Expectation
        stein_est = evaluate_stein_expectation(dist, dim,(-10,10), 300, h=h)
        
        langevin_est = eval_Langevin(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)
        hmc_est = eval_HMC(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)

        # since the moment sums over each dimension, the true moment is the sum of the moments for each dimension
        true_moment = true_moments[dims.index(dim)]

        print("Dimension: ", dim)
        print(f'True moment: {true_moment}, Stein estimate: {stein_est}, Langevin estimate: {langevin_est}, HMC estimate: {hmc_est}')

        stein_error = abs(true_moment - stein_est)
        langevin_error = abs(true_moment - langevin_est)
        hmc_error = abs(true_moment - hmc_est)

        stein_ests.append(stein_est)
        langevin_ests.append(langevin_est)
        hmc_ests.append(hmc_est)

        stein_errors.append(stein_error)
        langevin_errors.append(langevin_error)
        hmc_errors.append(hmc_error)

    # plot the results
    plt.figure()
    plt.plot(dims, stein_ests, label='Stein')
    plt.plot(dims, langevin_ests, label='Langevin')
    plt.plot(dims, hmc_ests, label='HMC')
    plt.plot(dims, [true_moment]*len(dims), label='True')
    plt.xlabel('Dimension')
    plt.ylabel('Estimated Moment')
    plt.legend(loc='best')
    plt.title('Estimated Moment vs. Dimension for N(0, 5*I_d)')

    #save figure
    plt.savefig('./plots/moment_comparison_dim_Gaussian.png')


exp_compare_dim_Gaussian()