import torch
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
from utils import * 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def stein_g(x, g, logp):
    """Compute the Stein operator of g for a given log probability function logp."""
    x.to(device)
    score = get_grad(logp(x).sum(), x)
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    return stein_val_batches

def train_network(net, optimizer, sample, normal_dist, h, epochs, verbose=True):
    for e in range(epochs):
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
