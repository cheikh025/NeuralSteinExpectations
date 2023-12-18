import torch
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
from utils import * 
from network import MLP
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def stein_g(x, g, logp):
    """Compute the Stein operator of g for a given log probability function logp."""
    x = x.to(device)
    score = get_grad(logp(x).sum(), x)
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    return stein_val_batches

def train_network(net, optimizer, sample, normal_dist, h, epochs, verbose=True):
    sample = sample.to(device)
    net = net.to(device)
    for e in tqdm(range(epochs), desc='Training '):
        optimizer.zero_grad()

        stein_val = stein_g(sample, net, normal_dist.log_prob)

        grad_s = get_grad(stein_val.sum(), sample)
        grad_h = get_grad(h(sample).sum(), sample)

        loss = torch.sum((grad_s - grad_h)**2)
        loss.backward()
        optimizer.step()
        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {loss.item()}')
    return net

def evaluate_stein_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000):
    # Initialize distribution and MLP network
    net = MLP(n_dims=net_dims, n_out=net_dims)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Generate and prepare sample data
    sample = dist.generate_points(n_samples, sample_range)
    sample.requires_grad = True

    # Train the network and estimate the moment
    trained_net = train_network(net, optimizer, sample, dist, h, epochs, verbose=False)
    est_moment = h(sample) 
    est_moment -= stein_g(sample, trained_net, dist.log_prob).to(est_moment.device)
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())
