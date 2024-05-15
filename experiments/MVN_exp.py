from distributions import *
from utils import *
from ToyDistributions import *
import torch
from neuralStein import *
from LangevinSampler import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
SEED = [1,3,42,25]
MEAN = 3
STD = np.sqrt(5)

Data = {'dim': [], 'seed': [], 'true_val': [], 'stein_est_diff': [], 
        'stein_est_grad': [], 'Langevin': [], 'HMC': []}
for dim in DIMS:
    for seed in SEED:
        print(f"dim: {dim}, seed: {seed}, True Val: {dim*(MEAN**2 + STD**2)}")
        torch.manual_seed(seed)
        dist = MultivariateNormalDistribution(mean=MEAN*torch.ones(dim).to(device), 
                                            covariance=(STD**2)*torch.eye(dim).to(device))
        LMC_est = eval_Langevin(dist, dim=dim, h=h, num_samples=100, 
                                num_chains=1024, device=device)
        print(f"\t Langevin est: {LMC_est.item()}")
        HMC_est = eval_HMC(dist, dim=dim, h=h, num_samples=100, 
                           num_chains=1024, device=device)
        print(f"\t HMC est: {HMC_est}")
        stein_est_grad = evaluate_stein_expectation(dist, dim, (-10,10), 300, h = h, 
                                                    epochs=1000, loss_type = "grad")
        print(f"\t Stein est grad: {stein_est_grad}")
        stein_est_diff = evaluate_stein_expectation(dist, dim, (-10,10), 300, h = h,
                                                    epochs=1000, loss_type = "diff")
        print(f"\t Stein est diff: {stein_est_diff}")
        Data['dim'].append(dim)
        Data['seed'].append(seed)
        Data['true_val'].append(dim*(MEAN**2 + STD**2))
        Data['Langevin'].append(LMC_est)
        Data['HMC'].append(HMC_est)
        Data['stein_est_diff'].append(stein_est_diff)
        Data['stein_est_grad'].append(stein_est_grad)

df = pd.DataFrame(Data)
df.to_csv("MVN_exp.csv")
Rsquared = {}
for method in ['Langevin', 'HMC', 'stein_est_diff', 'stein_est_grad']:
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
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.title(r'MVN Experiment $\mathcal{N}(3, 5*I_d)$')
plt.savefig('MVN_exp.png')
plt.show()