from distributions import *
from utils import *
from ToyDistributions import *
import torch
import torch.distributions as tdist

import pandas as pd
import seaborn as sns
import scipy 

from neuralStein import *
from LangevinSampler import *
from neural_CV import *
from control_functional import *

# set all seeds for numpy, and torch, and random
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# Set the aesthetic style of the plots
#sns.set(style="whitegrid", palette="pastel")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_points(n_samples,dim, sample_range=(-5, 5)):
    return torch.rand(n_samples, dim) * (sample_range[1] - sample_range[0]) + sample_range[0]

s_lim = 5
mb_size = 100

def identity(x):
    return x.sum(-1)

def square(x):
    return (x**2)

def square_sum(x):
    return (x**2).sum(dim=-1).unsqueeze(-1)

def interval_exp(dim=1, device=device, seed = 123, plot_true=False, n_points = 100):
    """
     dist = Gaussian 
     h = square 
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    h = square_sum

    unif_samples = generate_points(n_points, dim, sample_range=(-s_lim,s_lim)).to(device)

    MEAN  = 1.0
    STD = 1.0
    true_moment = (MEAN**2 + STD**2)*dim



    # set up distribution, a Multivariate Gaussian
    for dim in [dim]:
        dist = MultivariateNormalDistribution(mean = (MEAN+torch.zeros(dim)).to(device),
                                              covariance=((STD**2)*torch.eye(dim)).to(device)
                                              ) 
        true_samples = (MEAN + STD*torch.randn((n_points,dim))).to(device)#.reshape(-1, 1)
        true_samples.requires_grad = True

        
        # Evaluate Stein Expectation
        stein_est, net_nse = evaluate_stein_expectation(dist, dim, (-s_lim,s_lim), n_points, h=h, 
                                                        loss_type="grad", epochs=500,
                                                         given_sample = unif_samples, return_learned=True, mb_size = mb_size)
        
        # Evaluate Stein Expectation
        stein_est_diff, net_nse_diff = evaluate_stein_expectation(dist, dim, (-s_lim,s_lim), n_points, h=h, 
                                                        loss_type="diff", epochs=500,
                                                         given_sample = unif_samples, return_learned=True, mb_size = mb_size)
        
        # evaluate neural CV expectation 
        cv_est, net_ncv ,c_ncv  = evaluate_ncv_expectation(dist, dim, (-s_lim,s_lim), n_points, h=h, epochs=500, 
                                                           given_sample=unif_samples, return_learned=True, mb_size = mb_size)
        
        # use network trained on off-samples, but estimate expectation on true samples
        # should give better estimate than just cv_est
        ncv_true_sample_vals = stein_g(true_samples, net_ncv, dist.log_prob) + c_ncv
        ncv_off_train_on_est = (h(true_samples) - ncv_true_sample_vals).mean().item()

        ncv_on_Tg_mean = ncv_true_sample_vals.mean().item()


        # evaluate neural CV expectation 
        cv_on_sample_est, net_on_sample_ncv, c_on_sample_ncv  = evaluate_ncv_expectation(dist, dim, (-s_lim, s_lim), n_points, h=h, epochs=500, 
                                                                                         given_sample = true_samples, return_learned=True, mb_size = mb_size)

        

        # evaluate CF expectation 
        
        #cf_est, cf_obj = evaluate_cf_expectation(dist = dist, sample_range=(-s_lim,s_lim), 
        #                        n_samples= n_points, h = h, 
        #                        reg = 0., given_sample = unif_samples, 
        #                        tune_kernel_params = True, return_learned= True)
        #print("CF est: ", cf_est)
        cf_est = 0.
        
        print("Stein est: ", stein_est)
        print("CV est (off sample): {}, CV off train on Est: {}, CV Tg On Sample Mean: {}".format(cv_est, ncv_off_train_on_est, ncv_on_Tg_mean))
        print("CV estimate (on sample): ", cv_on_sample_est)
        
        
        print("True moment: ", true_moment) 
        

        if plot_true:
            x_plot = torch.linspace(-5,5,500).to(device).reshape(-1, 1)

            x_plot.requires_grad = True 
            
            # evaluate on x plot 
            stein_preds_plot = stein_g(x_plot, net_nse, dist.log_prob)

            cv_est_plot = stein_g(x_plot, net_ncv, dist.log_prob)
            cv_on_sample_est_plot = stein_g(x_plot, net_on_sample_ncv, dist.log_prob)

            score_x_plot = get_grad(dist.log_prob(x_plot).sum(), x_plot).detach()
            cf_preds_plot = cf_obj.pred_f(x_plot, score_x_plot)   

            plt.figure(figsize =(10,10))
            plt.plot(x_plot.detach().cpu().numpy(), cf_preds_plot.detach().cpu().numpy(), label = "CF (Est = {:.3f})".format(cf_est))        
            plt.plot(x_plot.detach().cpu().numpy(), stein_preds_plot.detach().cpu().numpy(), label = "Ours (Est = {:.3f})".format(stein_est))
            plt.plot(x_plot.detach().cpu().numpy(), cv_est_plot.detach().cpu().numpy(), label = "NCV (Est = {:.3f})".format(cv_est))
            plt.plot(x_plot.detach().cpu().numpy(), cv_on_sample_est_plot.detach().cpu().numpy(), label = "NCV, target samples (Est = {:.3f})".format(cv_on_sample_est))
        
        
            plt.ylim(-5,5)

            #plt.plot(x_plot.detach().cpu().numpy(), cf_preds_plot.detach().cpu().numpy(), label = "CF")
            plt.plot(x_plot.detach().cpu().numpy(), h(x_plot).detach().cpu().numpy(), label = "h(x) = x^2")
            plt.plot(x_plot.detach().cpu().numpy(), h(x_plot).detach().cpu().numpy() - true_moment, label = "Ground Truth h(x) - E[h(X)] = x^2 - 2.0", color='k')

            # plot the unif samples
            plt.scatter(unif_samples.detach().cpu().numpy(), [0.]*n_points, label = "Unif samples")
            plt.scatter(true_samples.detach().cpu().numpy(), [0.]*n_points, label = "True samples")

            plt.legend(loc='best')

            plt.savefig('plots/interval_exp_Tg_plot_sample_range_{}.png'.format(s_lim))


            # for all methods, plot g(x) (compared to true function)
            def true_soln(x):
                return -x-MEAN 
        
            def true_soln_c(x, c):
                c = c.detach().cpu().numpy()
                func = -x-MEAN + (2.-c)*np.sqrt(np.pi/2)*np.exp(0.5 * (x-MEAN)**2) * scipy.special.erf((x-MEAN)/np.sqrt(2))

                # add in extra term for k
                k = 4.2 
                func_k = func + k*np.exp(x**2 / 2 - x)
                return func_k

            x_plot_np = x_plot.detach().cpu().numpy()
            plt.figure(figsize = (10, 10))
            plt.ylim(-5,5)
            plt.plot(x_plot_np, true_soln(x_plot_np), label = "True g(x) soln = -x-1.", color='k')
            plt.plot(x_plot_np, net_nse(x_plot).detach().cpu().numpy(), label = "Ours g(x)")
            plt.plot(x_plot_np, net_ncv(x_plot).detach().cpu().numpy(), label = "NCV g(x)")
            plt.plot(x_plot_np, net_on_sample_ncv(x_plot).detach().cpu().numpy(), label = "NCV, trained on target samples g(x)")
            plt.plot(x_plot_np, true_soln_c(x_plot_np, c_ncv), label = "Soln with c = {:.3f}".format(c_ncv), color='k', linestyle='--')
            #plt.plot(x_plot_np, cf_obj.net(x_plot).detach().cpu().numpy(), label = "CF g(x)")
        
            plt.legend(loc='best')
        
            plt.savefig("plots/interval_exp_g_plot_sample_range_{}.png".format(s_lim))
        
        mc_est = h(true_samples).mean().item()

        return mc_est, cv_est, cv_on_sample_est,ncv_off_train_on_est, ncv_on_Tg_mean, cf_est, stein_est, stein_est_diff

mc_ests = []
cv_ests = []
cv_off_train_on_ests = []
cv_Tg_means = []
cv_on_sample_ests = []
cf_ests = []
stein_ests_grad = []
stein_ests_diff = []
true_moments = []

exp_tag = 'multiDim_gaussian_mu_{}_std_{}_range_{}_h_sqaured_mb_size_{}'.format(1, 1, s_lim, mb_size)

dims = [500, 750, 1000]
cur_dims = []
fixed_npoints = False #True

for dim in dims:
    for trial_seed in [11]:
        
        if fixed_npoints:
            n_points = 100
        else:
            n_points = max(100*int(dim/10), 100)

        print("\n\n Dim: {}\n".format(dim))
        mc_est, cv_est, cv_on_sample_est, ncv_off_train_on_est, ncv_on_Tg_mean, cf_est, stein_est, stein_est_diff = interval_exp(dim=dim, device=device, seed = trial_seed, plot_true=False, n_points=n_points)

        mc_ests.append(mc_est)
        cv_ests.append(cv_est)
        cv_off_train_on_ests.append(ncv_off_train_on_est)
        cv_Tg_means.append(ncv_on_Tg_mean)
        cv_on_sample_ests.append(cv_on_sample_est)
        cf_ests.append(cf_est)
        stein_ests_grad.append(stein_est)
        stein_ests_diff.append(stein_est_diff)
        true_moments.append(dim*2.)

        cur_dims.append(dim)

        # save intermediates
        dataDict = {'Dim': cur_dims, 'True': true_moments, 'MC': mc_ests, 'CV': cv_ests, 'CV offTrain onEst': cv_off_train_on_ests, 'CV E_p[Tg]': cv_Tg_means, 'CV_On_Sample': cv_on_sample_ests, 'CF': cf_ests, 'Stein_Grad': stein_ests_grad, 'Stein_Diff': stein_ests_diff}
        df = pd.DataFrame(dataDict)
        df.to_csv('./results/{}_500_on.csv'.format(exp_tag))



dataDict = {'Dim': dims, 'True': true_moments, 'MC': mc_ests, 'CV': cv_ests, 'CV offTrain onEst': cv_off_train_on_ests, 'CV E_p[Tg]': cv_Tg_means, 'CV_On_Sample': cv_on_sample_ests, 'CF': cf_ests, 'Stein_Grad': stein_ests_grad, 'Stein_Diff': stein_ests_diff}

df = pd.DataFrame(dataDict)
df.to_csv('./results/{}.csv'.format(exp_tag))

plt.rcParams.update({'axes.titlesize': 20})

plt.figure(figsize=(20,10))
plt.plot(dims, true_moments, label = "True", color='k', marker='o')
plt.plot(dims, mc_ests, label = "MC", marker='o')
plt.plot(dims, cv_ests, label = "CV", marker='o')
plt.plot(dims, cv_off_train_on_ests, label = "CV OffTrain OnEst", marker='o')
plt.plot(dims, cv_on_sample_ests, label = "CV OnSample", marker='o')
plt.plot(dims, cf_ests, label = "CF", marker='o')
plt.plot(dims, stein_ests_grad, label = "Stein Grad", marker='o')
plt.plot(dims, stein_ests_diff, label = "Stein Est Diff", marker='o')
plt.legend(loc='best')

# make boxplot for different methods, to illustrate variance
#ax = df.plot(kind='box', title='Estimates of E[X^2] for N(1, 1^2), True Val = 2.', showmeans=True, figsize=(20,10), fontsize=20)

plt.savefig('./plots/{}_500_on.png'.format(exp_tag))
