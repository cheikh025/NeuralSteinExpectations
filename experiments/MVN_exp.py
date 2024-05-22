import torch
import pandas as pd
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distributions import *
from utils import *
from ToyDistributions import *
from neural_CV import *
from control_functional import *
from neuralStein import *
from LangevinSampler import *

# Parameters
HOME = "experiments/MVN_results/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIMS = [1,2,3,5,10,15,30,50]
SEED = [13,17,23,42,169]
MEAN = 3
STD = np.sqrt(5)
sample_range = (-10,10)
EPOCHS = [1000, 1500, 1800, 2300, 3000, 4000, 5000, 7000]
N_SAMPLES = [300, 500, 800, 1200, 1500, 2000, 3000, 4000]

# Define the function h(x)
def h(x):
    # I changed this to sum along last axis (to allow multiple chains)
    #return torch.sum(x**2, dim=-1)
    dim = x.shape[-1]
    if dim == 1:
        return x**2
    return (x**2).sum(-1)

def get_parameters(experiment):
    seed = SEED[experiment % len(SEED)]
    dim = DIMS[experiment // len(SEED)]
    epochs = EPOCHS[experiment // len(SEED)]
    n_samples = N_SAMPLES[experiment // len(SEED)]
    return dim, seed, epochs, n_samples


def main(args):
    Data = {'dim': [], 'seed': [], 'true_val': [], 'NSE_diff': [], 
            'NSE_grad': [], 'Langevin': [], 'HMC': [],
            'CF': [], 'NCV': []}
    dim, seed, epochs, n_samples = get_parameters(args.experiment - 1)
    print(f"dim: {dim}, seed: {seed}, True Val: {dim*(MEAN**2 + STD**2)}")
    torch.manual_seed(seed)
    dist = MultivariateNormalDistribution(mean=MEAN*torch.ones(dim).to(device), 
                                        covariance=(STD**2)*torch.eye(dim).to(device))
    # true_samples = MEAN + STD*torch.randn(n_samples, dim).to(device)
    LMC_est = eval_Langevin(dist, dim=dim, h=h, num_samples=n_samples, 
                            num_chains=100, device=device)
    print(f"\t Langevin est: {LMC_est.item()}")
    HMC_est = eval_HMC(dist, dim=dim, h=h, num_samples=n_samples, 
                    num_chains=100, device=device)
    print(f"\t HMC est: {HMC_est}")
    NSE_grad = evaluate_stein_expectation(dist, dim, sample_range, n_samples, h = h, mb_size=1024,
                                                epochs=epochs, loss_type = "grad", resample_=True)
    print(f"\t NSE est grad: {NSE_grad}")
    NSE_diff = evaluate_stein_expectation(dist, dim, sample_range, n_samples, h = h, mb_size=1024,
                                        epochs=epochs, loss_type = "diff", resample_=True)
    print(f"\t NSE est diff: {NSE_diff}")
    cf_est = evaluate_cf_expectation(dist = dist, sample_range=sample_range,
                            n_samples= n_samples, h = h,
                            reg=0., given_sample = None)
    print(f"\t CF off-samples est: {cf_est}")
    # cf_on_est = evaluate_cf_expectation(dist = dist, sample_range=sample_range,
    #                         n_samples= n_samples, h = h,
    #                         reg=0., given_sample = true_samples)
    # print(f"\t CF on-samples est: {cf_on_est}")
    # ncv_est = evaluate_ncv_expectation(dist, dim, sample_range, n_samples, h, 
    #                                 epochs=epochs, reg = 0., resample_=True)
    # print(f"\t NCV off-samples est: {ncv_est}")
    # ncv_on_est = evaluate_ncv_expectation(dist, dim, sample_range, n_samples, h, 
    #                                 epochs=epochs, reg = 0., given_sample = true_samples)
    # print(f"\t NCV on-samples est: {ncv_on_est}")

    ncv_varg_est = evaluate_varg_expectation(dist, dim, sample_range, n_samples, h, mb_size=1024,
                                             epochs=epochs, reg = 0., resample_=True)
    print(f"\t NCV varg est: {ncv_varg_est}")
    # ncv_varg_on_est = evaluate_varg_expectation(dist, dim, sample_range, n_samples, h,
    #                                          epochs=epochs, reg = 0., given_sample = true_samples)
    # print(f"\t NCV varg on-samples est: {ncv_varg_on_est}")
    Data['dim'].append(dim)
    Data['seed'].append(seed)
    Data['true_val'].append(dim*(MEAN**2 + STD**2))
    Data['Langevin'].append(LMC_est)
    Data['HMC'].append(HMC_est)
    Data['NSE_diff'].append(NSE_diff)
    Data['NSE_grad'].append(NSE_grad)
    Data['CF'].append(cf_est)
    #Data['CF_on'].append(cf_on_est)
    Data['NCV'].append(ncv_varg_est)
    #Data['NCV_on'].append(ncv_on_est)
    df = pd.DataFrame(Data)
    print(Data)
    df.to_csv(HOME + f"MVN_exp_{args.experiment}.csv")


if __name__ == '__main__':    
    # To run this script, use the following command:
    # python MVN_exp.py --experiment=NO_OF_EXPERIMENT
    # NO_OF_EXPERIMENT is an integer from 1 to 32
    # Example : if NO_OF_EXPERIMENT = 1, then dim = 1, seed = 13, epochs = 1000, n_samples = 300 
    #           if NO_OF_EXPERIMENT = 5, then dim = 2, seed = 13, epochs = 1500, n_samples = 500
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    # Define the --experiment argument
    parser.add_argument('--experiment', type=int, required=False, default=1,
                        help='No of the experiment to run')
    
    # Parse the arguments
    args = parser.parse_args()

    main(args)
    


