import torch
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
from utils import * 
from network import MLP, normalizedMLP
from tqdm import tqdm

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stein_g(x, g, logp):
    """Compute the Stein operator of g for a given log probability function logp."""
    score = get_grad(logp(x).sum(), x)
    
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    
    # for 1d input, unpad to [N, 1]
    if x.shape[-1] == 1:
        return stein_val_batches.unsqueeze(1) 

    return stein_val_batches

def stein_g_precomputed_score(x, g, score_x):
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score_x).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    
    if x.shape[-1] == 1:
        return stein_val_batches.unsqueeze(1) 
    
    return stein_val_batches

# mb_size = None means no minibatching/full batch 
def train_network_grad_loss(net, optimizer, sample, target_dist, h, epochs, verbose=True, mb_size = None):

    # precompute h(sample) and logp(sample)
    h_sample = h(sample)

    #logp_sample = target_dist.log_prob(sample)

    # data minibatches
    # full batch 
    if mb_size is None:
        batch_idx = torch.arange(0,sample.size(0)).long()
        mb_size = sample.size(0)
    else:
        print("**Minibatches with batch size: ", mb_size)
        batch_idx = torch.randperm(sample.size(0))

    for e in tqdm(range(epochs), desc='Training '):
        epoch_loss = 0.

        # over minibatches
        for b_num in range(0, sample.size(0), mb_size):
            idx = batch_idx[b_num: b_num + mb_size]
            sample_mb = sample[idx, :]
            h_sample_mb = h_sample[idx]

            optimizer.zero_grad()

            stein_val = stein_g(sample_mb, net, target_dist.log_prob)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')

            assert(stein_val.shape == h_sample_mb.shape), f"Stein val shape: {stein_val.shape}, H sample shape: {h_sample.shape}"

            grad_s = get_grad(stein_val.sum(), sample_mb)
            grad_h = get_grad(h(sample_mb).sum(), sample_mb) # NOTE: preferably uses precomputed h_sample_mb, look into improving
        
            assert(grad_s.shape == grad_h.shape)

            loss = torch.sum((grad_s - grad_h)**2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()/sample.size(0)

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}')
    return net

# mb_size = None means no minibatching/full batch 
def train_network_diff_loss(net, optimizer, sample, target_dist, h, epochs, verbose=True, mb_size = None):
    device = sample.device

    #perturbed samples
    sample_bar = sample + (torch.randn_like(sample)).to(device)

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    h_sample_bar = h(sample_bar).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = get_grad(logp_sample.sum(), sample).detach()

    logp_sample_bar = target_dist.log_prob(sample_bar)
    score_sample_bar = get_grad(logp_sample_bar.sum(), sample_bar).detach()

    # data minibatches
    # full batch 
    if mb_size is None:
        batch_idx = torch.arange(0,sample.size(0)).long()
        mb_size = sample.size(0) 
    else:
        print("**Minibatches with batch size: ", mb_size)
        batch_idx = torch.randperm(sample.size(0))

    for e in tqdm(range(epochs), desc='Training '):
        epoch_loss = 0.

        for b_num in range(0, sample.size(0), mb_size):
            idx = batch_idx[b_num: b_num + mb_size]

            # get minibatch vals
            sample_mb = sample[idx, :]
            sample_bar_mb = sample_bar[idx,:]

            h_sample_mb = h_sample[idx]
            h_sample_bar_mb = h_sample_bar[idx]

            score_sample_mb = score_sample[idx,:]
            score_sample_bar_mb = score_sample_bar[idx,:]

            # loss and training             
            optimizer.zero_grad()

            stein_val = stein_g_precomputed_score(sample_mb, net, score_sample_mb)
            stein_val_bar = stein_g_precomputed_score(sample_bar_mb, net, score_sample_bar_mb)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape), f"Stein val shape: {stein_val.shape}, H sample shape: {h_sample_mb.shape}"
            assert(stein_val.device == h_sample_mb.device)

            loss = torch.mean(( (stein_val - h_sample_mb) - (stein_val_bar - h_sample_bar_mb))**2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()/sample.size(0)
            
        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}')
    return net


# mb_size = None means no minibatching/full batch 
def train_network_diff_loss_no_perturb(net, optimizer, sample, target_score, h, epochs, verbose=True, mb_size = None):
    device = sample.device

    #perturbed samples
    bar_idx = torch.randperm(sample.shape[0])
    sample_bar = (sample[bar_idx].detach()).to(device).requires_grad_(True)

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    h_sample_bar = h(sample_bar).detach().to(device)

    score_sample =  target_score.detach()

    score_sample_bar = target_score[bar_idx].detach()

    # data minibatches
    # full batch 
    if mb_size is None:
        batch_idx = torch.arange(0,sample.size(0)).long()
        mb_size = sample.size(0) 
    else:
        print("**Minibatches with batch size: ", mb_size)
        batch_idx = torch.randperm(sample.size(0))

    for e in tqdm(range(epochs), desc='Training '):
        epoch_loss = 0.

        for b_num in range(0, sample.size(0), mb_size):
            idx = batch_idx[b_num: b_num + mb_size]

            # get minibatch vals
            sample_mb = sample[idx, :]
            sample_bar_mb = sample_bar[idx,:]

            h_sample_mb = h_sample[idx]
            h_sample_bar_mb = h_sample_bar[idx]

            score_sample_mb = score_sample[idx,:]
            score_sample_bar_mb = score_sample_bar[idx,:]

            # loss and training             
            optimizer.zero_grad()

            stein_val = stein_g_precomputed_score(sample_mb, net, score_sample_mb)
            stein_val_bar = stein_g_precomputed_score(sample_bar_mb, net, score_sample_bar_mb)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape), f"Stein val shape: {stein_val.shape}, H sample shape: {h_sample_mb.shape}"
            assert(stein_val.device == h_sample_mb.device)

            loss = torch.mean(( (stein_val - h_sample_mb) - (stein_val_bar - h_sample_bar_mb))**2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()/sample.size(0)
            
        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}')
    return net


#loss type is either 'grad' or 'diff'
def evaluate_stein_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000, loss_type = "grad", given_sample = None, given_score = None, network="MLP", return_learned = False, mb_size = None):
    # Initialize distribution and MLP network
    if network == 'NormalizedMLP':
        net = normalizedMLP(n_dims=net_dims, n_out=net_dims)
    else :
        net = MLP(n_dims=net_dims, n_out=net_dims)
    
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    if given_sample is None:
        #   Generate and prepare sample data
        sample = dist.generate_points(n_samples, sample_range).to(device)
        sample.requires_grad = True
    else:
        # copy given samples, and set requires_grad to True
        sample = given_sample.clone().detach().to(device).requires_grad_(True)
    

    # Train the network and estimate the moment
    if loss_type == "grad":
        trained_net = train_network_grad_loss(net, optimizer, sample, dist, h, epochs, verbose=True, mb_size = mb_size)
    elif loss_type == "diff":
        if given_score is None:
            trained_net = train_network_diff_loss(net, optimizer, sample, dist, h, epochs, verbose=True, mb_size = mb_size)
        else:
            trained_net = train_network_diff_loss_no_perturb(net, optimizer, sample, given_score, h, epochs, verbose=True, mb_size = mb_size)

    est_moment = h(sample).detach() 

    if given_score is None:
        est_moment -= stein_g(sample, trained_net, dist.log_prob).to(est_moment.device).detach()
    else:
        est_moment -= stein_g_precomputed_score(sample, trained_net, given_score).to(est_moment.device).detach()
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")

    if return_learned:
        return est_moment.mean().item(), trained_net 
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())
