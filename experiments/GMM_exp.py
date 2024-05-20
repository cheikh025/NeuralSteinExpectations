import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

HOME = "experiments/GMM_results/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIMS = [1,2,3,5,10,20,30]
SEED = [7,13,23,42,169]
sample_range = (-6,6)
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

    LMC_est = eval_Langevin(dist, dim=dim, h=h, num_samples=n_samples, 
                            num_chains=100, device=device)
    print(f"\t Langevin est: {LMC_est.item()}")
    HMC_est = eval_HMC(dist, dim=dim, h=h, num_samples=n_samples, 
                    num_chains=100, device=device)
    print(f"\t HMC est: {HMC_est}")
    NSE_grad = evaluate_stein_expectation(dist, dim, sample_range, n_samples, h = h, 
                                                mb_size=512, epochs=epochs, loss_type = "grad")
    print(f"\t Stein est grad: {NSE_grad}")
    NSE_diff = evaluate_stein_expectation(dist, dim, sample_range, n_samples, mb_size=512,
                                                h = h, epochs=epochs, loss_type = "diff")
    print(f"\t Stein est diff: {NSE_diff}")
    cf_est = evaluate_cf_expectation(dist = dist, sample_range=sample_range,
                            n_samples= n_samples, h = h,
                            reg=0., given_sample = None)
    print(f"\t CF off-samples est: {cf_est}")
    ncv_est = evaluate_ncv_expectation(dist, dim, sample_range, n_samples, h, mb_size=512,
                                    epochs=epochs, reg = 0.)
    print(f"\t NCV off-samples est: {ncv_est}")

    Data['dim'].append(dim)
    Data['seed'].append(seed)
    Data['true_val'].append(0.0)
    Data['Langevin'].append(LMC_est)
    Data['HMC'].append(HMC_est)
    Data['NSE_diff'].append(NSE_diff)
    Data['NSE_grad'].append(NSE_grad)
    Data['CF'].append(cf_est)
    Data['NCV'].append(ncv_est)
    Data['Epochs'].append(epochs)
    Data['n_samples'].append(n_samples)
    print(Data)
    df = pd.DataFrame(Data)
    df.to_csv(HOME + f"GMM_exp_{args.experiment}.csv")


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
    








