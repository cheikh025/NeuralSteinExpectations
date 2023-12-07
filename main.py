from distributions import NormalDistribution, ExponentialDistribution, NormalDistributionKD, StudentsTDistribution, CustomDistribution
from network import MLP
from utils import stein_g, train_network
import torch.optim as optim
import torch
from torch.distributions import StudentT

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
    sample = generate_shuffled_samples(net_dims, sample_range, n_samples)
    sample.requires_grad = True

    # Train the network and estimate the moment
    trained_net = train_network(net, optimizer, sample, dist, h, epochs)
    est_moment = h(sample) - stein_g(sample, trained_net, dist.log_prob)
    print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {est_moment.mean().item()}")

# Define the function h(x)
def h(x):
    return torch.sum(x**2, dim=1)

# define  the score function
def score_function(x):
      return x ** 2 + 2 * x + 1

# Evaluate for different distributions and conditions
#evaluate_stein_expectation(NormalDistribution(mean=10.0, std=3.0), 1, (5, 15), 100)
#evaluate_stein_expectation(ExponentialDistribution(rate=1.0), 1, (1, 3), 100)
#evaluate_stein_expectation(NormalDistributionKD(mean=torch.tensor([0.0, 0.0]), covariance=torch.tensor([[1.0, 0.0], [0.0, 1.0]])), 2, (-1, 1), 100)
#evaluate_stein_expectation(NormalDistributionKD(mean=torch.tensor([0.0, 0.0, 0.0]), covariance=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])), 3, (-1, 1), 100)
#evaluate_stein_expectation(StudentsTDistribution(nu=3.0), 1, (-10, 10), 300)
evaluate_stein_expectation(CustomDistribution(score_function=score_function), 1, (0, 10), 300)
