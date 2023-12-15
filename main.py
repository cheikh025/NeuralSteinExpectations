from distributions import *
from network import MLP
from utils import *
from ToyDistributions import *
import torch.optim as optim
import torch
from torch.distributions import StudentT
import random
import math
from bayes_opt import BayesianOptimization
from OtherMethods import HamiltonianMCMC 

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
    sample.requires_grad = True

    # Train the network and estimate the moment
    trained_net = train_network(net, optimizer, sample, dist, h, epochs, verbose=False)
    est_moment = h(sample) - stein_g(sample, trained_net, dist.log_prob)
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    return -abs(est_moment.mean().item() - dist.second_moment())

# Define the function h(x)
def h(x):
    return torch.sum(x**2, dim=1)


def test_other_methods():
    ### Test examples for the other methods ###
    # Parameters
    # test the methods
    h = lambda x: torch.sum(x ** 2)
    dim = 2
    mean = 10 * torch.ones(dim)
    cov = 3 * torch.eye(dim)
    distribution = MultivariateNormalDistribution(mean=mean, covariance=cov)
    # Run the MCMC methods
    num_samples = 1000
    step_size = 0.1
    hamiltonian = HamiltonianMCMC(distribution.log_prob, h, dim, step_size, sampler='hmc')
    ham_expectation, H_time = hamiltonian.compute_expectation(num_samples)
    print(f'Hamiltonian expectation:  {ham_expectation}, time: {H_time}')
    hamiltonian = HamiltonianMCMC(distribution.log_prob, h, dim, step_size, sampler='nuts')
    ham_expectation, H_time = hamiltonian.compute_expectation(num_samples)
    print(f'NUTS expectation:  {ham_expectation}, time: {H_time}')
    hamiltonian = HamiltonianMCMC(distribution.log_prob, h, dim, step_size, sampler='rmhmc')
    ham_expectation, H_time = hamiltonian.compute_expectation(num_samples)
    print(f'RMHMC expectation:  {ham_expectation}, time: {H_time}')

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
#parameter_variations= [
#            {'mean': 0, 'std': 1},
#            {'mean': 5, 'std': 2},
#            {'mean': 10, 'std': 3},]
#for parameter in parameter_variations:
#    dist_instance = NormalDistribution(**parameter)
#                #best_range = find_best_range(dist_instance, 1)
#    Estimated = evaluate_stein_expectation(dist_instance, 1, (-100,100), 1000)
#    print(f"Estimated moment for {dist_instance.__class__.__name__}: {Estimated}")
#    print(dist_instance.second_moment())


def find_best_range_bayesopt(dist, dim, min_start=1, max_end=30, num_iterations=200):
    def evaluate_stein_expectation_bayesopt(x,y):
        sample_range = (x,y)
        return evaluate_stein_expectation(dist, dim, sample_range, 100, 300)
    pbounds = {'x': (min_start, max_end-1), 'y': (min_start+1, max_end)}
    optimizer = BayesianOptimization(
        f=evaluate_stein_expectation_bayesopt,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(n_iter=num_iterations)
    return optimizer.max['params']

dist = NormalDistribution(mean=0, std=1)
print(find_best_range_bayesopt(dist, 1))

test_other_methods()