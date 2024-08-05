import torch
import sys
import os
import argparse
import time
import torch.distributions as tdist

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
DIMS = [1,2,3,5]
SEED = [7,13,23]
sample_range = (-8,18)
EPOCHS = [2000*int(dim/10 + 1) for dim in DIMS]
N_SAMPLES = [1000 for dim in DIMS]

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
    print(f"dim: {dim}, seed: {seed}, True Val: {(0.2*-5 + 0.8*5)*dim}")
    torch.manual_seed(seed)
    dist = Mixture(comps=[tdist.MultivariateNormal(loc = -5*torch.ones(dim).to(device), 
                                            covariance_matrix=torch.eye(dim).to(device)),
                                tdist.MultivariateNormal(loc = 5*torch.ones(dim).to(device),
                                            covariance_matrix=torch.eye(dim).to(device)
                                            )],
                            pi=torch.tensor([0.2, 0.8]).to(device)
                            )
    mesh_dist = Mixture(comps=[tdist.MultivariateNormal(loc = -5*torch.ones(dim).to(device), 
                                            covariance_matrix=torch.eye(dim).to(device)),
                                tdist.MultivariateNormal(loc = 5*torch.ones(dim).to(device),
                                            covariance_matrix=torch.eye(dim).to(device)
                                            )],
                            pi=torch.tensor([0.5, 0.5]).to(device))
    samples = mesh_dist.sample(n_samples*2) #generate_points(n_samples=n_samples*2, sample_range=sample_range)
    # Generate a random permutation of indices
    perm = torch.randperm(samples.size(0))
    samples = samples[perm]
    #print(dist.sample(n_samples*2).mean())
    print(samples.shape, samples.mean(), samples.std())
    def plot_1_dim_MoG(dist, sample_range, samples):
        x = torch.linspace(sample_range[0], sample_range[1], 1000).reshape(-1,1).to(device)
        y = torch.exp(dist.log_prob(x)).cpu().numpy()
        plt.plot(x, y)
        plt.scatter(samples, torch.zeros_like(samples).cpu().numpy(), color='red')
        plt.show()
        
    # Plot the target distribution pdf (dist) and the samples
    plot_1_dim_MoG(dist, sample_range,  samples)
    start = time.time()
    # LMC_est = eval_Langevin(dist, dim=dim, h=h, num_samples=5000,
    #                         num_chains=100, device=device, gamma=0.01, alpha=0.95)
    # print(f"\t Langevin est: {LMC_est.item()}, time: {time.time() - start}")
    # start = time.time()
    # HMC_est = eval_HMC(dist, dim=dim, h=h, num_samples=5000, 
    #                 num_chains=100, device=device, alpha=0.05, num_L_steps=1)
    # print(f"\t HMC est: {HMC_est}, time: {time.time() - start}")
    start = time.time()
    NSE_grad, net = evaluate_stein_expectation(dist, dim, sample_range, n_samples, h = h, given_sample=samples,
                                        epochs=epochs, loss_type = "grad", resample_=False, return_learned=True)
    samples = samples.requires_grad_(True)
    s_vals = stein_g(samples, net, dist.log_prob).detach()
    moment_vals = h(samples)
    plt.figure(figsize= (15, 10))
    plt.scatter(samples.detach().numpy(), s_vals.detach().numpy(), label = "(Tg)(x)", marker='x')
    plt.scatter(samples.detach().numpy(), moment_vals.detach().numpy(), label  = "h(x)", marker='o')
    plt.xlabel("Points x_i")
    plt.title("h(x) vs (Tg)(x)")
    plt.legend(loc='best')
    plt.show()
    print(f"\t NSE est grad: {NSE_grad}, time: {time.time() - start}")
    start = time.time()
    NSE_diff, net = evaluate_stein_expectation(dist, dim, sample_range, n_samples, given_sample=samples,
                                        h = h, epochs=epochs, loss_type = "diff", resample_=False, return_learned=True)
    s_vals = stein_g(samples, net, dist.log_prob).detach()
    moment_vals = h(samples)
    plt.figure(figsize= (15, 10))
    plt.scatter(samples.detach().numpy(), s_vals.detach().numpy(), label = "(Tg)(x)", marker='x')
    plt.scatter(samples.detach().numpy(), moment_vals.detach().numpy(), label  = "h(x)", marker='o')
    plt.xlabel("Points x_i")
    plt.title("h(x) vs (Tg)(x)")
    plt.legend(loc='best')
    plt.show()
    print(f"\t NSE est diff: {NSE_diff}, time: {time.time() - start}")
    start = time.time()
    cf_est = evaluate_cf_expectation(dist = dist, sample_range=sample_range, given_sample=samples,
                            n_samples= n_samples, h = h,
                            reg=0.)
    print(f"\t CF off-samples est: {cf_est}, time: {time.time() - start}")
    start = time.time()
    ncv_est = evaluate_varg_expectation(dist, dim, sample_range, n_samples, h,  given_sample=samples,
                                             epochs=epochs, reg = 0., resample_=False)
    print(f"\t NCV off-samples est: {ncv_est}, time: {time.time() - start}")

    Data['dim'].append(dim)
    Data['seed'].append(seed)
    Data['true_val'].append(0.0)
    # Data['Langevin'].append(LMC_est)
    # Data['HMC'].append(HMC_est)
    Data['NSE_diff'].append(NSE_diff)
    #Data['NSE_grad'].append(NSE_grad)
    Data['CF'].append(cf_est)
    Data['NCV'].append(ncv_est)
    Data['Epochs'].append(epochs)
    Data['n_samples'].append(n_samples)
    print(Data)
    #df = pd.DataFrame(Data)
    #df.to_csv(HOME + f"GMM_exp_{args.experiment}.csv")


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    # Define the --experiment argument
    parser.add_argument('--experiment', type=int, required=False, default=2,
                        help='No of the experiment to run')
    
    # Parse the arguments
    args = parser.parse_args()

    main(args)
    








