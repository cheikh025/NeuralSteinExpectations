import torch
import numpy as np 
import seaborn as sns
import pandas as pd

import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distributions import * 
#from NeuralSteinExpectations.distributions import *

from neural_CV import *
from neuralStein import *
from control_functional import *

#seed = 23

# Set the random seed for reproducible results
#torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params_to_test = [1, 3, 5, 7]
params_sample_range = {1: (0, 2), 3: (0, 2), 5: (-5, 5), 7: (0, 2)}
seeds = [23, 2, 8 , 6, 10]
results = []
all_results = []
for PARAM_TO_ESTIMATE in params_to_test:
    estimates = {'NCV': [], 'Stein_diff': [], 'CF': [], 'MCOnMesh': [], 'MCTrue': [],'Stein_grad': []}
    for seed in seeds:
        torch.manual_seed(seed)
        #PARAM_TO_ESTIMATE = 3  #1 #7 # This gives the index of the parameter to estimate : 0 = x_1, 1 = x_2, 2 = x_3, 3 = x_4
        N = 1000
        EPOCHS = 3000

        def integrand(x):
            return x[:, PARAM_TO_ESTIMATE]

        # prior means and variances (on unconstrained parameters)
        p_means = torch.tensor([0, -3, -3, 0, np.log(10), np.log(10), -1, -1]).to(device).float()
        p_variances = torch.tensor([0.5**2, 0.5**2, 0.5**2, 0.5**2, 1**2, 1**2, 1**2, 1**2]).to(device).float()
        p_covs = torch.diag(p_variances)

        # we will always work with unconstrained parameters (the score is wrt to these)

        # for mesh, we can sample from prior or from uniform around parameter mean 
        #prior = torch.distributions.MultivariateNormal(p_means, p_covs)
        #prior = torch.distributions.MultivariateNormal(torch.Tensor([-2, -3., -1., -3., 3., 1., -1., -1.]).to(device) + torch.zeros_like(p_means, device=device), 
        #                                             torch.Tensor([1., 1., 1., 1., 1., 1., 1., 1.]).to(device) * torch.eye(p_means.shape[0],device=device))

        #prior = torch.distributions.MultivariateNormal(torch.Tensor([-3, -3., -3., -3., 3., 3., -3., -3.]).to(device) + torch.zeros_like(p_means, device=device), 
        #                                             torch.Tensor([1., 1., 1., 1., 1., 1., 1., 1.]).to(device) * torch.eye(p_means.shape[0],device=device))

        # works well for param 1, doesn't throw error often for some reason
        #prior =  torch.distributions.MultivariateNormal(-3.5 + torch.zeros_like(p_means, device=device), torch.eye(p_means.shape[0],device=device))

        # latest used (Apr 30 2024)

        center = torch.Tensor([-1, -3., -1., -3., 3., 1., -1., -1.]).to(device)
        prior = torch.distributions.Uniform(center - 2, center + 0.1*torch.Tensor([.1, .1, .1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(device))

        #prior_samples = prior.sample((N,)) # (N, 8) - full set of parameters (unconstrained)

        #torch.save(prior_samples, 'prior_samples_unc.pt')
        #exit() 

        prior_samples = torch.load('./ode_exps/data/prior_samples_unc.pt')
        prior_score = torch.load('./ode_exps/data/score_unc_prior.pt')

        dim = 8 #1 # change to 1 when using independence approx grad log p(x_i)
        sample_range = (p_means[PARAM_TO_ESTIMATE].cpu().detach().numpy()-5, p_means[PARAM_TO_ESTIMATE].cpu().detach().numpy()+5)

        # load data, with the precomputed score
        all_data = torch.load('./ode_exps/data/all_data_lv.pt')

        #unconstrained samples (4000, 8) - 4000 total samples and 8 parameters
        post_samples = all_data['X_all']

        # score at unconstrained samples
        post_score_samples = all_data['score_unconstrainedsamples_all']

        # mesh samples are subset of posterior samples
        idx =  torch.randperm(N)

        mesh_samples = prior_samples[idx,:] #[:, PARAM_TO_ESTIMATE] #post_samples[idx, :][:, PARAM_TO_ESTIMATE] # (N, 1) - 1024 samples
        mesh_score = prior_score[idx,:] #[:, PARAM_TO_ESTIMATE] #post_score_samples[idx, :][:, PARAM_TO_ESTIMATE] # (N, 1) - 1024 samples

        # for checking on policy 
        #mesh_samples = post_samples[idx, :][:, PARAM_TO_ESTIMATE] # (N, 1) - 1024 samples
        #mesh_score = post_score_samples[idx, :][:, PARAM_TO_ESTIMATE] # (N, 1) - 1024 samples


        mesh_samples = mesh_samples#.unsqueeze(1)
        mesh_score = mesh_score#.unsqueeze(1)

        print(mesh_samples.shape)

        stein_est = evaluate_stein_expectation(prior, 
                                   dim,
                                   sample_range= sample_range,#params_sample_range[PARAM_TO_ESTIMATE], 
                                   n_samples = N, 
                                   h =integrand,
                                   epochs=EPOCHS,
                                   loss_type = "diff",
                                   given_sample = mesh_samples.to(device),
                                   given_score = mesh_score.to(device))
        
        stein_grad = evaluate_stein_expectation(prior, 
                           dim,
                           sample_range= sample_range, 
                           n_samples = N, 
                           h =integrand,
                           epochs=EPOCHS,
                           loss_type = "grad",
                           given_sample = mesh_samples.to(device),
                           given_score = mesh_score.to(device))

        ncv_est = evaluate_ncv_expectation(prior, 
                                   dim,
                                   sample_range= sample_range,#params_sample_range[PARAM_TO_ESTIMATE],
                                   n_samples = N, 
                                   h =integrand,
                                   epochs=EPOCHS,
                                   reg = 0.,
                                   given_sample = mesh_samples.to(device),
                                   given_score = mesh_score.to(device))

        cf_est, cf_obj = evaluate_cf_expectation(dist = prior, sample_range=sample_range,#params_sample_range[PARAM_TO_ESTIMATE],
                                        n_samples= N, h = integrand,
                                        reg=0., given_sample = mesh_samples.to(device),
                                        given_score=mesh_score.to(device),
                                        tune_kernel_params = True, return_learned= True)

        print("Param {} True (MC) val: ".format(PARAM_TO_ESTIMATE), post_samples[:, PARAM_TO_ESTIMATE].mean())
        print("Param {} MC Estimate on Mesh data: ".format(PARAM_TO_ESTIMATE), mesh_samples.mean())
        print("Param {} NCV_est: ".format(PARAM_TO_ESTIMATE), ncv_est)
        print("Param {} (diff) Stein_est: ".format(PARAM_TO_ESTIMATE), stein_est)
        print("Param {} CF_est: ".format(PARAM_TO_ESTIMATE), cf_est)
        print("Param {} Stein_grad_est: ".format(PARAM_TO_ESTIMATE), stein_grad)

        estimates['NCV'].append(ncv_est)
        estimates['Stein_diff'].append(stein_est)
        estimates['CF'].append(cf_est)
        estimates['MCOnMesh'].append(mesh_samples.mean().cpu().detach().numpy())
        estimates['MCTrue'].append(post_samples[:, PARAM_TO_ESTIMATE].mean())
        estimates['Stein_grad'].append(stein_grad)
        mc_true = post_samples[:, PARAM_TO_ESTIMATE].mean()
        all_results.append({
            'Parameter': PARAM_TO_ESTIMATE,
            'Seed': seed,
            'Method': 'NCV',
            'Error': abs(ncv_est - mc_true)
        })
        all_results.append({
            'Parameter': PARAM_TO_ESTIMATE,
            'Seed': seed,
            'Method': 'Stein',
            'Error': abs(stein_est - mc_true)
        })
        all_results.append({
            'Parameter': PARAM_TO_ESTIMATE,
            'Seed': seed,
            'Method': 'CF',
            'Error': abs(cf_est - mc_true)
        })
        all_results.append({
            'Parameter': PARAM_TO_ESTIMATE,
            'Seed': seed,
            'Method': 'MCOnMesh',
            'Error': abs(mesh_samples.mean() - mc_true)
        })
        all_results.append({
            'Parameter': PARAM_TO_ESTIMATE,
            'Seed': seed,
            'Method': 'Stein_grad',
            'Error': abs(stein_grad - mc_true)
        })
    
    avg_estimates = {k: np.mean(v) for k, v in estimates.items()}
    results.append(avg_estimates)
df_results = pd.DataFrame(results, index=[f"Parameter {p}" for p in params_to_test])
df_results.to_csv('./ode_exps/data/lv_ode_results.csv')

df_all_results = pd.DataFrame(all_results)
df_all_results.to_csv('./ode_exps/data/lv_ode_errors.csv')

data = pd.read_csv('./ode_exps/data/lv_ode_errors.csv')
data['Error'] = data['Error'].str.extract(r'(\d+\.\d+)').astype(float)


# Plotting boxplot for the errors of each parameter and method using seed values 23, 46, 57 to see the distribution of errors
plt.figure(figsize=(10, 6))
sns.boxplot(x='Parameter', y='Error', hue='Method', data=data)
plt.xlabel('Parameter')
plt.ylabel('Error')
plt.title('Error of each parameter and method')
plt.savefig('./ode_exps/lv_ode_errors.png')