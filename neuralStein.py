import torch
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
from utils import * 
from network import MLP, normalizedMLP
from tqdm import tqdm

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def stein_g(x, g, logp):
    device = x.device

    """Compute the Stein operator of g for a given log probability function logp."""
    x = x.to(device)
    score = get_grad(logp(x).sum(), x)
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    
    # for 1d input, unpad to [N, 1]
    if len(stein_val_batches.shape) == 1:
        return stein_val_batches.unsqueeze(1) 

    return stein_val_batches

def stein_g_precomputed_score(x, g, score_x):
    device = x.device

    x = x.to(device)
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score_x).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    
    if len(stein_val_batches.shape) == 1:
        return stein_val_batches.unsqueeze(1) 
    
    return stein_val_batches

def train_network_grad_loss(net, optimizer, sample, target_dist, h, epochs, verbose=True):
    device = sample.device
    
    sample = sample.to(device)
    net = net.to(device)

    # precompute h(sample) and logp(sample)
    h_sample = h(sample)

    #logp_sample = target_dist.log_prob(sample)

    for e in tqdm(range(epochs), desc='Training '):
        optimizer.zero_grad()

        stein_val = stein_g(sample, net, target_dist.log_prob)

        #print(f'Stein val shape: {stein_val.shape}')
        #print(f'H sample shape: {h_sample.shape}')

        assert(stein_val.shape == h_sample.shape), "Stein val shape: {stein_val.shape}, H sample shape: {h_sample.shape}"

        grad_s = get_grad(stein_val.sum(), sample)
        grad_h = get_grad(h_sample.sum(), sample)
        
        assert(grad_s.shape == grad_h.shape)

        loss = torch.sum((grad_s - grad_h)**2)
        loss.backward()
        optimizer.step()
        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {loss.item()}')
    return net

def train_network_diff_loss(net, optimizer, sample, target_dist, h, epochs, verbose=True):
    device = sample.device
    
    sample = sample.to(device)

    #perturbed samples
    sample_bar = (sample + torch.randn_like(sample)).to(device)

    net = net.to(device)

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    h_sample_bar = h(sample_bar).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = get_grad(logp_sample.sum(), sample).detach()

    logp_sample_bar = target_dist.log_prob(sample_bar)
    score_sample_bar = get_grad(logp_sample_bar.sum(), sample_bar).detach()

    for e in tqdm(range(epochs), desc='Training '):
        optimizer.zero_grad()

        stein_val = stein_g_precomputed_score(sample, net, score_sample)
        stein_val_bar = stein_g_precomputed_score(sample_bar, net, score_sample_bar)

        #print(f'Stein val shape: {stein_val.shape}')
        #print(f'H sample shape: {h_sample.shape}')
              
        assert(stein_val.shape == h_sample.shape), f"Stein val shape: {stein_val.shape}, H sample shape: {h_sample.shape}"
        assert(stein_val.device == h_sample.device)

        loss = torch.mean(( (stein_val - h_sample) - (stein_val_bar - h_sample_bar))**2)
        loss.backward()
        optimizer.step()
        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {loss.item()}')
    return net


#loss type is either 'grad' or 'diff'
def evaluate_stein_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000, loss_type = "grad", given_sample = None, network="MLP", return_learned = False):
    # Initialize distribution and MLP network
    if network == 'NormalizedMLP':
        net = normalizedMLP(n_dims=net_dims, n_out=net_dims)
    else :
        net = MLP(n_dims=net_dims, n_out=net_dims)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    if given_sample is None:
        #   Generate and prepare sample data
        sample = dist.generate_points(n_samples, sample_range)
        sample.requires_grad = True
    else:
        # copy given samples, and set requires_grad to True
        sample = given_sample.clone().detach().requires_grad_(True)
    sample.to(device)

    # Train the network and estimate the moment
    if loss_type == "grad":
        trained_net = train_network_grad_loss(net, optimizer, sample, dist, h, epochs, verbose=True)
    elif loss_type == "diff":
        trained_net = train_network_diff_loss(net, optimizer, sample, dist, h, epochs, verbose=True)

    est_moment = h(sample).detach() 
    est_moment -= stein_g(sample, trained_net, dist.log_prob).to(est_moment.device).detach()
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")

    if return_learned:
        return est_moment.mean().item(), trained_net 
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())
