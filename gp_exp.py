import random 
from distributions import *
from utils import *
from ToyDistributions import *
import torch
import torch.distributions as tdist
from neuralStein import *
from LangevinSampler import *
import pandas as pd
import seaborn as sns
from neural_CV import *
from control_functional import *

# portions of code copied from Sarcos experiments at https://github.com/jz-fun/Meta_Control_Variates/tree/main/Sarcos

# Set the aesthetic style of the plots
sns.set(style="whitegrid", palette="pastel")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GP Kernel
def base_kernel(X1, X2, kerparms):
    if len(X1.size()) == 1:  #: as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
        X1 = X1.unsqueeze(1)
    if len(X2.size()) == 1:
        X2 = X2.unsqueeze(1)

    kernel_parm1 = kerparms[0]
    kernel_parm2 = kerparms[1]

    dist_mat = torch.cdist(X1, X2, p=2) ** 2

    m = X1.size()[0]
    n = X2.size()[0]


    norms_X1 = X1.norm(dim=1, p=2).pow(2)  # as we assume each row represents a point, we compute norm by rows.
    norms_X2 = X2.norm(dim=1, p=2).pow(2)

    norms_X1 = norms_X1.unsqueeze(dim=1)  # size is [m,1]
    norms_X2 = norms_X2.unsqueeze(dim=0)  # size is [1,n]

    mat = (1 + kernel_parm1 * norms_X1.repeat(1, n)) * (1 + kernel_parm1 * norms_X2.repeat(m, 1))

    prior_covariance = (1 / (mat)) * torch.exp(-0.5 * dist_mat / kernel_parm2 ** 2)
    return prior_covariance


# Sarcos robot arm data
X = torch.load('X.pt')
y = torch.load('y.pt')

Xstar = torch.load('Xstar.pt')
Ystar = torch.load('ystar.pt')


# variational approx over log kernel parameters (eta) for GP
# prior is x_1 ~ Gamma(25, 25), x_2 ~ Gamma(25, 25)
# mapped to eta_1, eta_2 = log(x_1), log(x_2)
# learned parameters for mu and cov of Normal distribution variational approx
post_mean_etaparms = torch.tensor([-0.1824,  0.1950]).to(device)
post_cov_etaparms = torch.tensor([[ 0.0029, -0.0025], [-0.0025,  0.0065]]).to(device)
post_dist = tdist.MultivariateNormal(post_mean_etaparms, post_cov_etaparms)
post_samples = post_dist.sample((3000,))


dim = 2

# log prob of variational approx distribution over kernel parameters 
def log_prob(x):
    return tdist.MultivariateNormal(post_mean_etaparms, post_cov_etaparms).log_prob(x)

# For GP, compute using a subset of data Nprime
N = 1000
li = list(range(0, X.size()[0]))
random.seed(0)
sub_idces = random.sample(li, N)

X_sub = X[sub_idces].to(device)
y_sub = y[sub_idces].to(device)

Nprime = 100
lisub = list(range(0, N))
random.seed(0)
subsub_idces = random.sample(lisub, Nprime)
Xprime = X_sub[subsub_idces].to(device)

# Set the testing point which indexes the integrand
# idx of target point, we treat ystar as an unbiased estimator for predictive mean at xstar
idx = 2
xstar = X[idx].squeeze().unsqueeze(0).to(device) # (1, d) (d=27 here - d for SARCOS dataset)
ystar = y[idx].squeeze().to(device)

# h(x) for this problem, we want to estimate the expectation of h(x) wrt the posterior
# shape of X is (n, 2) where n is the number of samples
def integrand(eta):
    theta= eta.exp()
    n = theta.size()[0]
    f_vals = torch.zeros(n,1)
    for i in range(n):
        K_star_Nprime = base_kernel(xstar, Xprime, theta[i])
        K_Nprime = base_kernel(Xprime, Xprime,theta[i])
        K_Nprime_N = base_kernel(Xprime, X_sub, theta[i])
        K_N_Nprime = K_Nprime_N.t()
        sigma = 0.1

        # ALT 1 - use solve to compute inverse * vector product
        tmp_val = torch.linalg.solve(A = (K_Nprime_N @ K_N_Nprime + sigma**2 * K_Nprime),  B = (K_Nprime_N @ y_sub))
        f_vals[i, :] = (K_star_Nprime @ tmp_val).squeeze() 
       

        # ALT 2 - maybe computing inverse like this directly is more expensive
        #f_vals[i,:] = (K_star_Nprime @ torch.inverse(K_Nprime_N @ K_N_Nprime + sigma**2 * K_Nprime) @ (K_Nprime_N @ y_sub)).squeeze()
    return f_vals

def integrand_mean(eta):
    return eta[:, 0].unsqueeze(-1)

# we have h, and log prob, so just run regular code to estimate expected value  

# its expensive to compute h for each sample, so precompute it

# target dist 
dist = MultivariateNormalDistribution(mean = post_mean_etaparms, covariance = post_cov_etaparms)


n_sample = 1024 #1024
n_epoch = 1500


# set seeds
torch.manual_seed(12)
np.random.seed(12)

f_vals_true_samples = integrand(post_samples).detach().cpu().numpy()
print("h(true samples eta): ", f_vals_true_samples)
post_samples_est = f_vals_true_samples.mean()

off_samples = dist.generate_points(n_sample, (0,2)).to(device)
off_samples_est = integrand(off_samples).mean().item()

# the training mesh is uniformly sampled from hypercube [-10, 10]^dim, n_samples total
stein_est = evaluate_stein_expectation(dist, 
                           dim,
                           sample_range= (0,2), 
                           n_samples = n_sample, 
                           h =integrand,
                           epochs=n_epoch,
                           loss_type = "diff")
print(f'Analytic true moment: {ystar}, True Sampled est: {post_samples_est}, Stein estimate diff: {stein_est}, Diff: {np.abs(stein_est - post_samples_est)}')

stein_est_given_samples = evaluate_stein_expectation(dist, 
                           dim,
                           sample_range= (0,2), 
                           n_samples = n_sample, 
                           h =integrand,
                           epochs=n_epoch,
                           loss_type = "diff",
                           given_sample = post_samples[:n_sample].to(device))
print(f'Analytic true moment: {ystar}, True Sampled est: {post_samples_est}, Stein estimate diff (using ON-samples): {stein_est_given_samples}, Diff: {np.abs(stein_est_given_samples - post_samples_est)}')

stein_est_grad = evaluate_stein_expectation(dist, 
                           dim,
                           sample_range= (0,2), 
                           n_samples = n_sample, 
                           h =integrand,
                           epochs=n_epoch,
                           loss_type = "grad")
print(f'Analytic true moment: {ystar}, True Sampled est: {post_samples_est}, Stein Est Grad: {stein_est_grad}, Diff: {np.abs(stein_est_grad - post_samples_est)}')

# compare to NCV
ncv_est_given_samples = evaluate_ncv_expectation(dist, 
                           dim,
                           sample_range= (0,2), 
                           n_samples = n_sample, 
                           h =integrand,
                           epochs=n_epoch,
                           given_sample = post_samples[:n_sample].to(device))  
print(f'NCV on sample: ', ncv_est_given_samples)

ncv_est = evaluate_ncv_expectation(dist, 
                           dim,
                           sample_range= (0,2), 
                           n_samples = n_sample, 
                           h =integrand,
                           epochs=n_epoch)               


print(f'Analytic true moment: {ystar}, MC Sampled est: {post_samples_est}, NCV estimate (using off-samples): {ncv_est}, Diff: {np.abs(ncv_est - post_samples_est)}')

# Control Functional
#cf_est, cf_obj = evaluate_cf_expectation(dist = dist, sample_range=(0,2),
#                                n_samples= n_sample, h = integrand,
#                                reg=0., given_sample = None,
#                                tune_kernel_params = True, return_learned= True)

# compare to Langevin and HMC
#eval_Langevin(dist = dist, dim = dim, h=integrand, num_samples=10, num_chains=100)

#eval_HMC(dist = dist, dim = dim, h=integrand, num_samples=10, num_chains=100)

f_vals_true_samples = integrand(post_samples).detach().cpu().numpy()
print("h(true samples eta): ", f_vals_true_samples)
post_samples_est = f_vals_true_samples.mean()

diff_stein_diff = np.abs(stein_est - post_samples_est)
print(f'Analytic true moment: {ystar}, True Sampled est: {post_samples_est}, Stein estimate diff: {stein_est}, Diff: {np.abs(stein_est - post_samples_est)}')

diff_stein_grad = np.abs(stein_est_grad - post_samples_est)
print(f'Analytic true moment: {ystar}, True Sampled est: {post_samples_est}, Stein Est Grad: {stein_est_grad}, Diff: {np.abs(stein_est_grad - post_samples_est)}')

diff_ncv = np.abs(ncv_est - post_samples_est)
print(f'Analytic true moment: {ystar}, MC Sampled est: {post_samples_est}, NCV estimate (using off-samples): {ncv_est}, Diff: {np.abs(ncv_est - post_samples_est)}, Off sample est: {off_samples_est}')
print(f'NCV on sample: ', ncv_est_given_samples)

# save gp results to csv file
# load existing gp results file
res_name = 'gp_results_samples_{}_epochs_{}.csv'.format(n_sample, n_epoch)
try:
    df = pd.read_csv('./results/GP_Results/{}'.format(res_name))
except:
    df = pd.DataFrame()

results_dict = {'xstar_idx': idx,'ystar': ystar.item(), 'post_samples_est': post_samples_est, 'unif(0,2)_off sample_est': off_samples_est, 'stein_est_diff': stein_est, 'stein_est_grad': stein_est_grad, 'ncv_est': ncv_est, 'ncv_on_sample_est': ncv_est_given_samples, 'diff_stein_diff': diff_stein_diff, 'diff_stein_grad': diff_stein_grad, 'diff_ncv': diff_ncv}
results = pd.DataFrame([results_dict])
df = pd.concat([df, results])
df.to_csv('./results/GP_Results/{}'.format(res_name), index=False)

#print(f'Analytic true moment: {ystar}, MC Sampled est: {post_samples_est}')


#print(f'Analytic true moment: {ystar}, MC Sampled est: {post_samples_est} \n NCV estimate (true samples): {ncv_est_given_samples}, NCV estimate (using off-samples): {ncv_est} \n Stein estimate (true samples): {stein_est_given_samples}, Stein estimate (using off-samples): {stein_est}')
#print(f'Analytic true moment: {ystar}, MC Sampled est: {post_samples_est} \n CF estimate: {cf_est}')