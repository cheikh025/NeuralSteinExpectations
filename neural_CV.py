import torch
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
from utils import * 
from network import MLP
from tqdm import tqdm
from neuralStein import stein_g, stein_g_precomputed_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_network_ncv_loss(net, c, optimizer, optim_c, sample, target_dist, h, epochs, reg = 0., verbose=True, mb_size = None):
    sample = sample.to(device)

    net = net.to(device)
    c = c.to(device)

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = get_grad(logp_sample.sum(), sample).detach()
    
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
            h_sample_mb = h_sample[idx]

            optimizer.zero_grad()
            optim_c.zero_grad()

            stein_val = stein_g_precomputed_score(sample_mb, net, score_sample)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape)
            assert(stein_val.device == h_sample_mb.device)

            loss = torch.mean((stein_val + c - h_sample_mb)**2)

            if reg > 0.:
                loss += reg * (net(sample_mb)**2).mean()

            loss.backward()
            optimizer.step()
            optim_c.step() 
            
            epoch_loss += loss.item()/sample.size(0)

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, c value: {c.item()}')
    return net, c


def evaluate_ncv_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000, reg = 0., given_sample = None, return_learned = False, mb_size = None):
    # Initialize distribution and MLP network
    net = MLP(n_dims=net_dims, n_out=net_dims)
    net = torch.nn.DataParallel(net)
    c = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optim_c = torch.optim.Adam([c], lr=1e-3)

    if given_sample is None:
        #   Generate and prepare sample data
        sample = dist.generate_points(n_samples, sample_range)
        sample.requires_grad = True
    else:
        # copy given samples, and set requires_grad to True
        sample = given_sample.clone().detach().requires_grad_(True)

    # Train the network and estimate the moment
    trained_net, trained_c = train_network_ncv_loss(net,c, optimizer, optim_c, sample, dist, h, epochs, reg=reg, verbose=True, mb_size = mb_size)

    h_sample = h(sample).detach() 
    est_moment = (h_sample - stein_g(sample, trained_net, dist.log_prob).to(h_sample.device).detach() + trained_c.detach()).mean()
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    print(f"Est moment NCV: {est_moment}, trained_c val: {trained_c.item()}")

    if return_learned:
        return est_moment.mean().item(), trained_net, trained_c 
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())
