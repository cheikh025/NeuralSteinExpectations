from distributions import *
from network import MLP
from utils import stein_g, train_network, expectation_sum_of_squares
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

# define  the score function
def score_function(x):
      return x ** 2 + 2 * x + 1

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

def generate_parameter_variations():
    parameter_variations= {
        NormalDistribution: [
            {'mean': 0, 'std': 1},
            {'mean': 5, 'std': 2},
            {'mean': 10, 'std': 3},
        ],
        ExponentialDistribution: [
            {'rate': 0.5},
            {'rate': 1.0},
            {'rate': 2.5},
        ],
        StudentsTDistribution: [
            {'nu': 2.5},
            {'nu': 5.0},
            {'nu': 10.0},
        ],
        LogisticDistribution: [
            {'mu': 0, 's': 1},
            {'mu': 5, 's': 2},
            {'mu': 10, 's': 3},
        ],
        KumaraswamyDistribution: [
            {'a': 0.5, 'b': 0.5},
            {'a': 1.0, 'b': 1.0},
            {'a': 2.0, 'b': 2.0},
        ],
        GammaDistribution: [
            {'alpha': 1.0, 'beta': 0.5},
            {'alpha': 2.0, 'beta': 1.0},
            {'alpha': 3.0, 'beta': 1.5},
        ],
        LaplaceDistribution: [
            {'mu': 0, 'b': 1},
            {'mu': 5, 'b': 2},
            {'mu': 10, 'b': 3},
        ],
        BetaDistribution: [
            {'alpha': 0.5, 'beta': 0.5},
            {'alpha': 1.0, 'beta': 1.0},
            {'alpha': 2.0, 'beta': 2.0},
        ],
        ParetoDistribution: [
            {'alpha': 2.5, 'xm': 0.5},
            {'alpha': 10.0, 'xm': 1.0},
            {'alpha': 21.0, 'xm': 2.0},
        ],
        WeibullDistribution: [
            {'k': 0.5, 'l': 0.5},
            {'k': 1.0, 'l': 1.0},
            {'k': 2., 'l': 2.0},
        ],
        GumbelDistribution: [
            {'mu': 0, 'b': 1},
            {'mu': 5, 'b': 2},
            {'mu': 10, 'b': 3},
        ],
    }
    return parameter_variations



def generate_distribution_params(distribution_type, dim):
    if distribution_type == "DirichletDistribution":
        # alpha should be a positive vector
        alpha = torch.rand(dim) + 0.01  # Adding 0.1 to ensure positivity
        return {'alpha': alpha}

    elif distribution_type == "MultivariateTDistribution":
        # Generate mean (mu) and covariance
        mu = torch.rand(dim)
        A = torch.rand(dim, dim)
        covariance = torch.mm(A, A.t())
        return {'mu': mu, 'covariance': covariance}

    elif distribution_type == "MultinomialDistribution":
        # 'n' is a scalar, 'p' is a probability vector that sums to 1
        n = torch.randint(1, 10, (1,)).item()  # A random integer
        p = torch.rand(dim)
        p = p / p.sum()  # Normalize to sum to 1
        return {'n': n, 'p': p}

    elif distribution_type == "VonMisesFisherDistribution":
        # 'mu' is a unit vector, 'kappa' is a concentration parameter
        mu = torch.rand(dim)
        mu = mu / torch.norm(mu)  # Normalize to unit length
        kappa = torch.rand(1).item()
        return {'mu': mu, 'kappa': kappa}

    elif distribution_type == "MultivariateNormalDistribution":
        # Generate mean (mu) and covariance
        mu = torch.rand(dim)
        A = torch.rand(dim, dim)
        covariance = torch.mm(A, A.t())
        return {'mean': mu, 'covariance': covariance}

    else:
        raise ValueError("Unsupported distribution type")

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
def evaluate_all_multivariate_distributions():
    Distribution = [DirichletDistribution, MultivariateTDistribution, MultinomialDistribution, VonMisesFisherDistribution, MultivariateNormalDistribution]
    all_evaluations = []
    for dist_class in Distribution:
        dim = random.randint(2, 10)
        Estimated = create_and_evaluate(dist_class, dim)
        all_evaluations.append((dist_class.__name__, dim, Estimated))
    return all_evaluations
