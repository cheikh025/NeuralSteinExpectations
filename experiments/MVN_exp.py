import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distributions import *
from utils import *
from ToyDistributions import *
from neural_CV import *
from control_functional import *
from neuralStein import *
from LangevinSampler import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the function h(x)
def h(x):
    # I changed this to sum along last axis (to allow multiple chains)
    #return torch.sum(x**2, dim=-1)
    dim = x.shape[-1]
    if dim == 1:
        return x**2
    return (x**2).sum(-1)

def compute_Rsquared(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_mean = np.mean(y_true)
    SS_tot = np.sum((y_true - y_mean)**2)
    SS_res = np.sum((y_true - y_pred)**2)
    return 1 - SS_res/SS_tot


DIMS = [1,2,3,5,15,20,25,30,50]
SEED = [13,17,23,42]
MEAN = 3
STD = np.sqrt(5)
sample_range = (-10,10)
EPOCHS = [1000, 1500, 1800, 2000, 3000, 4000, 4000, 5000, 5000]
N_SAMPLES = [300, 500, 800, 1000, 1000, 1500, 1500, 2000, 2000]

Data = {'dim': [], 'seed': [], 'true_val': [], 'stein_est_diff': [], 
        'stein_est_grad': [], 'Langevin': [], 'HMC': [], 'CF': [], 'NCV': [], 'NCV_on': []}

for n_samples, epochs, dim in zip(N_SAMPLES, EPOCHS, DIMS):
    for seed in SEED:
        print(f"dim: {dim}, seed: {seed}, True Val: {dim*(MEAN**2 + STD**2)}")
        torch.manual_seed(seed)
        dist = MultivariateNormalDistribution(mean=MEAN*torch.ones(dim).to(device), 
                                            covariance=(STD**2)*torch.eye(dim).to(device))
        true_samples = MEAN + STD*torch.randn(n_samples, dim).to(device)
        LMC_est = eval_Langevin(dist, dim=dim, h=h, num_samples=n_samples, 
                                num_chains=1024, device=device)
        print(f"\t Langevin est: {LMC_est.item()}")
        HMC_est = eval_HMC(dist, dim=dim, h=h, num_samples=n_samples, 
                           num_chains=1024, device=device)
        print(f"\t HMC est: {HMC_est}")
        stein_est_grad = evaluate_stein_expectation(dist, dim, sample_range, n_samples, h = h, 
                                                    epochs=epochs, loss_type = "grad")
        print(f"\t Stein est grad: {stein_est_grad}")
        stein_est_diff = evaluate_stein_expectation(dist, dim, sample_range,
                                                    n_samples, h = h, epochs=epochs, loss_type = "diff")
        print(f"\t Stein est diff: {stein_est_diff}")
        cf_est, _ = evaluate_cf_expectation(dist = dist, sample_range=sample_range,
                                n_samples= n_samples, h = h,
                                reg=0., given_sample = None,
                                tune_kernel_params = True, return_learned= True)
        print(f"\t CF off-samples est: {cf_est}")
        cf_on_est, _ = evaluate_cf_expectation(dist = dist, sample_range=sample_range,
                                n_samples= n_samples, h = h,
                                reg=0., given_sample = true_samples)
        print(f"\t CF on-samples est: {cf_on_est}")
        ncv_est = evaluate_ncv_expectation(dist, dim, sample_range, n_samples, h, 
                                           epochs=1000, reg = 0.)
        print(f"\t NCV off-samples est: {ncv_est}")
        ncv_on_est = evaluate_ncv_expectation(dist, dim, sample_range, n_samples, h, 
                                        epochs=1000, reg = 0., given_sample = true_samples)
        print(f"\t NCV on-samples est: {ncv_on_est}")

        print(f"\t Stein est diff: {stein_est_diff}")
        Data['dim'].append(dim)
        Data['seed'].append(seed)
        Data['true_val'].append(dim*(MEAN**2 + STD**2))
        Data['Langevin'].append(LMC_est)
        Data['HMC'].append(HMC_est)
        Data['stein_est_diff'].append(stein_est_diff)
        Data['stein_est_grad'].append(stein_est_grad)
        Data['CF'].append(cf_est)
        Data['CF_on'].append(cf_on_est)
        Data['NCV'].append(ncv_est)
        Data['NCV_on'].append(ncv_on_est)

df = pd.DataFrame(Data)
df.to_csv("MVN_exp.csv")
Rsquared = {}
for method in ['Langevin', 'HMC', 'stein_est_diff', 'stein_est_grad', 
               'CF', 'CF_on', 'NCV', 'NCV_on']:
    Rsquared[method] = compute_Rsquared(df['true_val'], df[method])

plt.figure()
sns.lineplot(data=df, x='dim', y='true_val', label='True Val', marker='o')
sns.lineplot(data=df, x='dim', y='Langevin', 
             label=rf"LMC $R^2$ = {Rsquared['Langevin']:.4f}", marker='v')
sns.lineplot(data=df, x='dim', y='HMC', 
             label=rf"HMC $R^2$ = {Rsquared['HMC']:.4f}", marker='s')
sns.lineplot(data=df, x='dim', y='stein_est_diff', 
            label=rf"NSE Diff $R^2$ = {Rsquared['stein_est_diff']:.4f}", marker='p')
sns.lineplot(data=df, x='dim', y='stein_est_grad', 
             label=rf"NSE Grad $R^2$ = {Rsquared['stein_est_grad']:.4f}", marker='*')
sns.lineplot(data=df, x='dim', y='CF',
                label=rf"CF $R^2$ = {Rsquared['CF']:.4f}", marker='d')
sns.lineplot(data=df, x='dim', y='CF_on',
                label=rf"CF On $R^2$ = {Rsquared['CF_on']:.4f}", marker='D')
sns.lineplot(data=df, x='dim', y='NCV',
                label=rf"NCV Off $R^2$ = {Rsquared['NCV']:.4f}", marker='x')
sns.lineplot(data=df, x='dim', y='NCV_on',
                label=rf"NCV On $R^2$ = {Rsquared['NCV_on']:.4f}", marker='X')

plt.xlabel('Dimension')
plt.ylabel('Value')
plt.title(r'MVN Experiment $\mathcal{N}(3, 5*I_d)$')
plt.savefig('MVN_exp.png')
plt.show()