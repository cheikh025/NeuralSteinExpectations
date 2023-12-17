from distributions import *
from utils import *
from ToyDistributions import *
import torch
import torch.distributions as tdist
from neuralStein import *
from LangevinSampler import *
import pandas as pd
import seaborn as sns

def square(x):
            return (x**2).sum(-1)

def identity(x):
    return x.sum(-1)

def exp_compare_dim_Gaussian():
    h = square

    dims = [1, 2, 3, 5, 10, 50]
    MEAN  = 3.0
    STD = 5.0

    true_moments = [(MEAN**2 + STD**2)*dim for dim in dims]

    stein_ests = []
    langevin_ests = []
    hmc_ests = []   

    stein_errors = []
    langevin_errors = []
    hmc_errors = []

    # set up distribution, a Multivariate Gaussian
    for dim in dims:
        dist = MultivariateNormalDistribution(mean = MEAN+torch.zeros(dim),
                                              covariance=(STD**2)*torch.eye(dim)
                                              ) 
        #torch.distributions.MultivariateNormal(MEAN+torch.zeros(dim), (STD**2)*torch.eye(dim))

        # Evaluate Stein Expectation
        stein_est = evaluate_stein_expectation(dist, dim,(-10,10), dim*300, h=h, epochs=1000*(int(dim/10)+1))
        
        langevin_est = eval_Langevin(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)
        hmc_est = eval_HMC(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100)

        # since the moment sums over each dimension, the true moment is the sum of the moments for each dimension
        true_moment = true_moments[dims.index(dim)]

        print("Dimension: ", dim)
        print(f'True moment: {true_moment}, Stein estimate: {stein_est}, Langevin estimate: {langevin_est}, HMC estimate: {hmc_est}')

        stein_error = abs(true_moment - stein_est)
        langevin_error = abs(true_moment - langevin_est)
        hmc_error = abs(true_moment - hmc_est)

        stein_ests.append(stein_est)
        langevin_ests.append(langevin_est)
        hmc_ests.append(hmc_est)

        stein_errors.append(stein_error)
        langevin_errors.append(langevin_error)
        hmc_errors.append(hmc_error)

    # plot the results
    plt.figure()
    plt.plot(dims, stein_ests, label='Stein')
    plt.plot(dims, langevin_ests, label='Langevin')
    plt.plot(dims, hmc_ests, label='HMC')
    plt.plot(dims, true_moments, label='True')
    plt.xlabel('Dimension')
    plt.ylabel('Estimated Moment')
    plt.legend(loc='best')
    plt.title('Estimated Moment vs. Dimension for N(3, 5*I_d)')

    #save figure
    plt.savefig('moment_comparison_dim_Gaussian.png')

def exp_compare_over_multiple_distributions():
    h = identity
    dim = 2
    n_iter = 15
    dim_ho = 1
    # set up distributions, MVN, MOG and HO
    def harmonic_oscillator(x):
        return  harmonic_oscillator_distribution_log(x, alpha=torch.tensor([1, 1]), beta=1.0)
    dists = {"HO" : CustomDistribution(harmonic_oscillator, dim_ho),
            "MVN" : MultivariateNormalDistribution(mean = torch.zeros(dim),
                                            covariance=(3**2)*torch.eye(dim)
                                            ), 
            "MOG" : Mixture(comps=[MultivariateNormalDistribution(mean = -2*torch.ones(dim),
                                            covariance=(2**2)*torch.eye(dim)),
                                MultivariateNormalDistribution(mean = 2*torch.ones(dim),
                                            covariance=(2**2)*torch.eye(dim)
                                            )],
                            pi=torch.tensor([0.5, 0.5])
                            )
            }
    
    # Evaluate Stein Expectation
    estimations = []
    for i in range(n_iter):
        print(f"Iteration: {i+1}")
        for name, dist in dists.items():
            # Evaluate estimations for both samplers
            if name == "HO":
                dim_ = dim_ho
            else :
                dim_ = dim
            hmc_est = eval_HMC(dist=dist, dim=dim_, h=h, num_samples=10, num_chains=100)
            langevin_est = eval_Langevin(dist=dist, dim=dim_, h=h, num_samples=10, num_chains=100)
            stein_est = evaluate_stein_expectation(dist, dim_,(-10,10), dim*300, h=h, epochs=1000)

            # Store each estimation in the estimations list
            estimations.append({'distribution': name, 'sampler': 'hmc', 'estimation': hmc_est})
            estimations.append({'distribution': name, 'sampler': 'lmc', 'estimation': langevin_est})
            estimations.append({'distribution': name, 'sampler': 'stein', 'estimation': stein_est})
            print(f"\tDistribution: {name}, HMC: {hmc_est}, LMC: {langevin_est}, Stein: {stein_est}")
    
    data = pd.DataFrame(estimations)
    data.to_csv("estimations.csv", index=False)
    # Plot the results

    # Set the aesthetic style of the plots
    sns.set(style="whitegrid", palette="pastel")
    # Set up the matplotlib figure
    plt.figure(figsize=(20, 12))

    # Create a boxplot
    sns.boxplot(x='distribution', y='estimation', hue='sampler', data=data)

    # Add some labels and a title
    plt.xlabel('Distribution')
    plt.ylabel('Estimation Value')
    plt.title('Boxplot of Estimations by Distribution and Sampler')

    # Display the plot
    plt.show()
    


#exp_compare_dim_Gaussian()
exp_compare_over_multiple_distributions()