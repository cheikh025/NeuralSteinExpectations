import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import argparse

# plot with x-axis - iterations/wall-clock time
# y-axis - variance of estimate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distributions import *
from utils import *
from ToyDistributions import *
from neural_CV import *
from control_functional import *
from neuralStein import *
from LangevinSampler import *

HOME = "./var_time_results/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIMS = [5] #[1,2,3,5,10,20,30]
SEED = [7]#,13,23,42,169]
sample_range = (-4,6)
EPOCHS = [1000*int(dim/10 + 1) for dim in DIMS]
N_SAMPLES = [300 for dim in DIMS]

# Define the function h(x)
def h(x):
    if x.shape[-1] == 1:
        return x
    return x.sum(-1)

def get_parameters(experiment):
    seed = SEED[experiment % len(SEED)]
    dim = DIMS[experiment // len(SEED)]
    epochs = EPOCHS[experiment // len(SEED)]
    n_samples = N_SAMPLES[experiment // len(SEED)]
    return dim, seed, epochs, n_samples


def main(args):
    Data = {'dim': [], 'seed': [], 'true_val': [], 'NSE_diff': [], 
            'NSE_grad': [], 'Langevin': [], 'HMC': [],
            'CF': [], 'NCV': [], 'Epochs': [], 'n_samples': []}
    dim, seed, epochs, n_samples = get_parameters(args.experiment - 1)
    print(f"dim: {dim}, seed: {seed}, True Val: 0.0")
    torch.manual_seed(seed)
    dist = Mixture(comps=[MultivariateNormalDistribution(mean = -2*torch.ones(dim).to(device),
                                            covariance=torch.eye(dim).to(device)),
                                MultivariateNormalDistribution(mean = 2*torch.ones(dim).to(device),
                                            covariance=torch.eye(dim).to(device)
                                            )],
                            pi=torch.tensor([0.5, 0.5]).to(device)
                            )

    #beh_dist = MultivariateNormalDistribution(mean = 1+torch.zeros(dim).to(device),
    #                                        covariance=10*torch.eye(dim).to(device))
    given_samples =  given_samples = sample_range[0]+(sample_range[1] - sample_range[0])*torch.rand(n_samples, dim) #beh_dist.sample((n_samples,))
     #beh_dist.sample((n_samples,))
    
    
    lmc_samples, lmc_sample_means, lmc_sample_vars, lmc_iter_times = eval_Langevin(dist, dim=dim, h=h, num_samples=n_samples, 
                            num_chains=100, var_time =True, init_samples = given_samples[:100, :], device=device)
    print(f"\t Langevin est: {lmc_sample_means[-1].item()}")
    LMC_est = lmc_sample_means[-1].item()

    if dim >= 2:
        plt.figure()
        plt.scatter(lmc_samples.reshape(-1, dim)[:,0], lmc_samples.reshape(-1, dim)[:,1], label="lmc samples")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("Scatter plot of LMC samples, Marginal along x1, x2")
        plt.savefig(HOME+'LMC_samples_2d.png')    

    hmc_samples, hmc_sample_means, hmc_sample_vars, hmc_iter_times = eval_HMC(dist, dim=dim, h=h, num_samples=n_samples, 
                    num_chains=100,  var_time = True, init_samples = given_samples[:100, :], device=device)
    print(f"\t HMC est: {hmc_sample_means[-1].item()}")
    HMC_est = hmc_sample_means[-1].item()
    
    if dim >= 2:
        plt.figure()
        plt.scatter(lmc_samples.reshape(-1, dim)[:,0], lmc_samples.reshape(-1, dim)[:,1], label="lmc samples")
        plt.scatter(hmc_samples.reshape(-1, dim)[:,0], hmc_samples.reshape(-1, dim)[:,1], label="hmc samples")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("Scatter plot of MCMC samples, Marginal along x1, x2")
        plt.savefig(HOME+'LMC_HMC_samples_2d.png') 
    
    NSE_grad, nseg_iter_times, nseg_est_iter = evaluate_stein_expectation(dist, dim, sample_range, n_samples, h = h, mb_size=1024, 
                                        epochs=epochs, loss_type = "grad", resample_=True, given_sample=None, var_time=True)
    print(f"\t NSE est grad: {NSE_grad}")
    
    NSE_diff, nsed_iter_times, nsed_est_iter = evaluate_stein_expectation(dist, dim, sample_range, n_samples, mb_size=1024,
                                        h = h, epochs=epochs, loss_type = "diff", resample_=True, given_sample = None, var_time=True)
    print(f"\t NSE est diff: {NSE_diff}")
    
    #cf_est = evaluate_cf_expectation(dist = dist, sample_range=sample_range,
    #                        n_samples= n_samples, h = h,
    #                        reg=0., given_sample = None)
    #print(f"\t CF off-samples est: {cf_est}")
    
    #ncv_est = evaluate_varg_expectation(dist, dim, sample_range, n_samples, h, mb_size=1024,
    #                                         epochs=epochs, reg = 0., resample_=True)
    #print(f"\t NCV off-samples est: {ncv_est}")
    
    
    plt.figure()
    plt.plot(lmc_iter_times, lmc_sample_vars, label='Langevin')
    plt.plot(hmc_iter_times, hmc_sample_vars, label='HMC')
    plt.xlabel('Wall-clock time')
    plt.ylabel('Variance of estimate')
    plt.legend()
    plt.title(f"dim={dim}, seed={seed} Var vs Wall-clock time")
    plt.savefig(HOME + f"var_time_exp_{args.experiment}.png")

    # plot means from mcmc
    plt.figure()
    plt.plot(lmc_iter_times, lmc_sample_means, label='Langevin')
    plt.plot(hmc_iter_times, hmc_sample_means, label='HMC')
    plt.plot(nseg_iter_times, nseg_est_iter, label='NSE_grad')
    plt.plot(nsed_iter_times, nsed_est_iter, label='NSE_diff')
    plt.plot(hmc_iter_times, [0.0]*len(hmc_iter_times), label='True Value')
    plt.xlabel('Wall-clock time')
    plt.ylabel('Estimate')
    plt.legend()
    plt.title(f"dim={dim}, seed={seed} Estimate vs Wall-clock time")
    plt.savefig(HOME + f"est_time_exp_{args.experiment}.png")


    Data['dim'].append(dim)
    Data['seed'].append(seed)
    Data['true_val'].append(0.0)
    Data['Langevin'].append(LMC_est)
    Data['HMC'].append(HMC_est)
    
    #Data['NSE_diff'].append(NSE_diff)
    #Data['NSE_grad'].append(NSE_grad)
    #Data['CF'].append(cf_est)
    #Data['NCV'].append(ncv_est)
    
    Data['lmc_iter_times'].append(lmc_iter_times)
    Data['hmc_iter_times'].append(hmc_iter_times)   

    Data['Epochs'].append(epochs)
    Data['n_samples'].append(n_samples)
    print(Data)
    df = pd.DataFrame(Data)
    df.to_csv(HOME + f"var_times_exp_{args.experiment}.csv")


if __name__ == '__main__':    
    # To run this script, use the following command:
    # python MVN_exp.py --experiment=NO_OF_EXPERIMENT
    # NO_OF_EXPERIMENT is an integer from 1 to 32
    # Example : if NO_OF_EXPERIMENT = 1, then dim = 1, seed = 42, epochs = 1000, n_samples = 300 
    #           if NO_OF_EXPERIMENT = 6, then dim = 2, seed = 42, epochs = 1000, n_samples = 500
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    # Define the --experiment argument
    parser.add_argument('--experiment', type=int, required=False, default=1,
                        help='No of the experiment to run')
    
    # Parse the arguments
    args = parser.parse_args()

    main(args)
    








