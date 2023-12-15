from distributions import *
from network import MLP
from utils import *
from ToyDistributions import *
import torch.optim as optim
import torch
from torch.distributions import StudentT
import random
import math
from OtherMethods import LangevinMCMC, HamiltonianMCMC 

def generate_shuffled_samples(dim, sample_range, n_samples):
    samples = []
    for _ in range(dim):
        sample = torch.linspace(*sample_range, n_samples)
        shuffled_indices = torch.randperm(sample.nelement())
        samples.append(sample.view(-1)[shuffled_indices].view(sample.size()))

    return torch.stack(samples, dim=1)

def evaluate_stein_expectation(dist, net_dims, sample_range, n_samples, epochs=1000):
    # Initialize distribution and MLP network
    net = MLP(n_dims=net_dims, n_out=net_dims)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Generate and prepare sample data
    sample = dist.generate_points(n_samples, sample_range)
    print(sample.shape)
    sample.requires_grad = True

    # Train the network and estimate the moment
    trained_net = train_network(net, optimizer, sample, dist, h, epochs)
    est_moment = h(sample) - stein_g(sample, trained_net, dist.log_prob)
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    return est_moment.mean().item()

# Define the function h(x)
def h(x):
    return torch.sum(x**2, dim=1)


def test_other_methods():
    ### Test examples for the other methods ###
    # Parameters
    dim = 1
    mean = 10 * torch.ones(dim)
    cov = 3 * torch.eye(dim)
    target_dist = {'name': 'gaussian', 'mean': mean, 'cov': cov}
    target_function = lambda x: torch.sum(x ** 2)
    num_samples = 100
    step_size = None
    target_accept = 0.9

    # Run the MCMC methods
    langevin = LangevinMCMC(target_dist, target_function, dim, step_size)
    lan_expectation, L_time = langevin.compute_expectation(num_samples)
    hamiltonian = HamiltonianMCMC(target_dist, target_function, dim, target_accept)
    ham_expectation, H_time = hamiltonian.compute_expectation(num_samples)
    print("Estimated expectation with Langevin MCMC method:", lan_expectation)
    print("Time taken for Langevin method:", L_time)
    print("Estimated expectation with Hamiltonian MCMC method:", ham_expectation)
    print("Time taken for Hamiltonian method:", H_time)



def find_best_range(dist, dim, min_start=-30, max_end=30, num_iterations=200):
    best_range = None
    best_performance = float('inf')

    for _ in range(num_iterations):
        min_range = random.randint(min_start, max_end - 1)
        max_range = random.randint(min_range + 1, max_end)
        
        sample_range = (min_range, max_range)
        performance = evaluate_stein_expectation(dist, dim, sample_range, 100, 300)

        if performance < best_performance:
            best_performance = performance
            best_range = sample_range
    print(f"Best range: {best_range}, best performance: {best_performance}")
    return best_range

def create_and_evaluate(distribution_class, dim):
     params = generate_distribution_params(distribution_class.__name__, dim)
     dist_instance = distribution_class(**params)
     #best_range = find_best_range(dist_instance, 1)
     Estimated = evaluate_stein_expectation(dist_instance, dim,(-2,2), 300)
     print(f"Estimated moment for {dist_instance.__class__.__name__}: {Estimated}")
     #true_estimated = expectation_sum_of_squares(dist_instance.mean, dist_instance.covariance)
     #print(f"True moment for {dist_instance.__class__.__name__}: {true_estimated}")
     return Estimated


def evaluate_all_univariate_distributions():
    parameter_variations = generate_parameter_variations()
    all_evaluations = []
    for dist_class, params_list in parameter_variations.items():
        for params in params_list:
            dist_instance = dist_class(**params)
            #best_range = find_best_range(dist_instance, 1)
            Estimated = evaluate_stein_expectation(dist_instance, 1, (-5,5), 300)
            print(f"Estimated moment for {dist_instance.__class__.__name__}: {Estimated}")
            all_evaluations.append((dist_class.__name__, params, Estimated))
    return all_evaluations
def evaluate_all_multivariate_distributions(dim):
    Distribution = [DirichletDistribution, MultivariateTDistribution, MultinomialDistribution, VonMisesFisherDistribution, MultivariateNormalDistribution]
    all_evaluations = []
    for dist_class in Distribution:
        Estimated = create_and_evaluate(dist_class, dim)
        all_evaluations.append((dist_class.__name__, dim, Estimated))
    return all_evaluations
parameter_variations= [
            {'mean': 0, 'std': 1},
            {'mean': 5, 'std': 2},
            {'mean': 10, 'std': 3},]
for parameter in parameter_variations:
    dist_instance = NormalDistribution(**parameter)
                #best_range = find_best_range(dist_instance, 1)
    Estimated = evaluate_stein_expectation(dist_instance, 1, (-100,100), 1000)
    print(f"Estimated moment for {dist_instance.__class__.__name__}: {Estimated}")
    print(dist_instance.second_moment())
