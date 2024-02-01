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
sns.set(style="whitegrid", palette="pastel")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_points(n_samples,dim, sample_range=(-5, 5)):
    return torch.rand(n_samples, dim) * (sample_range[1] - sample_range[0]) + sample_range[0]

s_lim = 5

n_points = 100

def identity(x):
    return x.sum(-1)

def square(x):
    return (x**2)

def interval_exp(dim=1, device=device):
    """
     dist = Gaussian 
     h = square 
    """
    h = square

    unif_samples = generate_points(n_points, dim, sample_range=(-s_lim,s_lim))

    MEAN  = 1.0
    STD = 1.0
    true_moment = (MEAN**2 + STD**2)*dim

    x_plot = torch.linspace(-5,5,500).to(device).reshape(-1, 1)
    x_plot.requires_grad = True 

    # set up distribution, a Multivariate Gaussian
    for dim in [dim]:
        dist = MultivariateNormalDistribution(mean = (MEAN+torch.zeros(dim)).to(device),
                                              covariance=((STD**2)*torch.eye(dim)).to(device)
                                              ) 
        true_samples = (MEAN + STD*torch.randn((n_points,))).to(device).reshape(-1, 1)

        
        # Evaluate Stein Expectation
        stein_est, net_nse = evaluate_stein_expectation(dist, dim, (-s_lim,s_lim), n_points, h=h, 
                                                        loss_type="grad", epochs=500,
                                                         given_sample = unif_samples, return_learned=True)
        
        # evaluate neural CV expectation 
        cv_est, net_ncv ,c_ncv  = evaluate_ncv_expectation(dist, dim, (-s_lim,s_lim), n_points, h=h, epochs=500, 
                                                           given_sample=unif_samples, return_learned=True)
        
        # evaluate neural CV expectation 
        cv_on_sample_est, net_on_sample_ncv, c_on_sample_ncv  = evaluate_ncv_expectation(dist, dim, (-s_lim, s_lim), n_points, h=h, epochs=500, 
                                                                                         given_sample = true_samples, return_learned=True)

        

        # evaluate CF expectation 
        
        cf_est, cf_obj = evaluate_cf_expectation(dist = dist, sample_range=(-s_lim,s_lim), 
                                n_samples= n_points, h = h, 
                                reg = 0., given_sample = unif_samples, 
                                tune_kernel_params = True, return_learned= True)
        print("CF est: ", cf_est)
        
        
        print("Stein est: ", stein_est)
        print("CV est (off sample): ", cv_est)
        print("CV estimate (on sample): ", cv_on_sample_est)
        
        
        print("True moment: ", true_moment)

        
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

interval_exp(dim=1, device=device)