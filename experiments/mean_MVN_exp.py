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

HOME = "results/mean_MVN_results/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIMS = [5]
SEED = [13,17,23,42]
MEAN = 3
MEANS = np.linspace(1, 10, 10)
STD = np.sqrt(5)
sample_range = (-10,10)
EPOCHS = [1500]
N_SAMPLES = [500]

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
            'NSE_grad': [], 'CF_on': [], 'NCV_on': []}
    dim, seed, epochs, n_samples = get_parameters(args.experiment - 1)
    print(f"dim: {dim}, seed: {seed}, True Val: {dim*(MEAN**2 + STD**2)}")
    torch.manual_seed(seed)
    dist = MultivariateNormalDistribution(mean=MEAN*torch.ones(dim).to(device), 
                                        covariance=(STD**2)*torch.eye(dim).to(device))
    for mean in MEANS:
        print(f"Sampling mean: {mean}")
        samples = mean + STD*torch.randn(n_samples, dim).to(device)
        
        NSE_grad = evaluate_stein_expectation(dist, dim, None, n_samples, h = h, 
                                                    given_sample=samples, epochs=epochs, loss_type = "grad")
        print(f"\t Stein est grad: {NSE_grad}")
        NSE_diff = evaluate_stein_expectation(dist, dim, None,  n_samples, 
                                                given_sample=samples, h = h, epochs=epochs, loss_type = "diff")
        print(f"\t Stein est diff: {NSE_diff}")
        cf_on_est = evaluate_cf_expectation(dist = dist, sample_range=None,
                                n_samples= n_samples, h = h,
                                reg=0., given_sample = samples)
        print(f"\t CF on-samples est: {cf_on_est}")
        ncv_on_est = evaluate_ncv_expectation(dist, dim, sample_range=None, n_samples=n_samples, h=h, 
                                        epochs=epochs, reg = 0., given_sample = samples)
        print(f"\t NCV on-samples est: {ncv_on_est}")

        print(f"\t Stein est diff: {NSE_diff}")
        Data['dim'].append(dim)
        Data['seed'].append(seed)
        Data['true_val'].append(dim*(MEAN**2 + STD**2))
        Data['NSE_diff'].append(NSE_diff)
        Data['NSE_grad'].append(NSE_grad)
        Data['CF_on'].append(cf_on_est)
        Data['NCV_on'].append(ncv_on_est)
        Data['Epochs'].append(epochs)
        Data['n_samples'].append(n_samples)
        Data['used_mean'].append(mean)
    df = pd.DataFrame(Data)
    df.to_csv(HOME + f"mean_MVN_exp_{args.experiment}.csv")


if __name__ == '__main__':    
    # To run this script, use the following command:
    # python MVN_exp.py --experiment=NO_OF_EXPERIMENT
    # NO_OF_EXPERIMENT is the number of the experiment to run
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    # Define the --experiment argument
    parser.add_argument('--experiment', type=int, required=False, default=1,
                        help='No of the experiment to run')
    
    # Parse the arguments
    args = parser.parse_args()

    main(args)
    


