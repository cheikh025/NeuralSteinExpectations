import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
from utils import * 
from network import MLP
from tqdm import tqdm
from neuralStein import stein_g, stein_g_precomputed_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def est_moment(stein_val, h_sample):
    return ((h_sample - stein_val)).mean()

def train_network_ncv_loss(net, c, optimizer, optim_c, sample, target_dist, h, epochs, reg = 0., 
                           verbose=True, mb_size = None, resample_ = False, sample_range = None):

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

        # initialize c properly
        if e == 0:
            c.data = est_moment(stein_g_precomputed_score(sample, net, score_sample), h_sample)

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

        if resample_:
            sample = target_dist.generate_points(sample.size(0), sample_range).to(device).requires_grad_(True)
            h_sample = h(sample).detach().to(device)
            logp_sample = target_dist.log_prob(sample)
            score_sample = get_grad(logp_sample.sum(), sample).detach()

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, c value: {c.item()}, est_moment: {est_moment(stein_g_precomputed_score(sample, net, score_sample), h_sample)}')
    return net, c



def train_network_ncv_loss_precomputed_score(net, c, optimizer, optim_c, sample, target_score, h, epochs, reg = 0., verbose=True, mb_size = None):

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    score_sample = target_score
    
    # data minibatches
    # full batch 
    if mb_size is None:
        batch_idx = torch.arange(0,sample.size(0)).long()
        mb_size = sample.size(0)
    else:
        print("**Minibatches with batch size: ", mb_size)
        batch_idx = torch.randperm(sample.size(0))

    for e in tqdm(range(epochs), desc='Training '):

        # initialize c properly
        if e == 0:
            c.data = est_moment(stein_g_precomputed_score(sample, net, score_sample), h_sample)

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
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, c value: {c.item()}, est_moment: {est_moment(stein_g_precomputed_score(sample, net, score_sample), h_sample)}')
    return net, c


# train with var grad:
# train with var min objective - ie. pass gradient through mean as well
# no c this time 
def train_network_var_min(net, optimizer, sample, target_dist, h, epochs, reg = 0., verbose=True, mb_size = None, 
                          with_perturb_samples = False, resample_ = False, sample_range = None):

    # if want to compare with same sample set as neural Stein
    if with_perturb_samples:
        samples_bar = (sample + torch.randn_like(sample)).to(device)
        sample = torch.cat([sample, samples_bar], dim = 0)

    losses = []
    grad_norms = []
    est_moments = []

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

            stein_val = stein_g_precomputed_score(sample_mb, net, score_sample)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape)
            assert(stein_val.device == h_sample_mb.device)

            # need to be differentiable wrt g here in mean as well
            # this is the key difference from the "train g" then "train c" alternatingly method above
            est_mean = (stein_val - h_sample_mb).mean()

            # empirical variance (note: NOT unbiased variance estimator)
            loss = torch.mean((stein_val - h_sample_mb - est_mean)**2)

            if reg > 0.:
                loss += reg * (net(sample_mb)**2).mean()

            loss.backward()
            optimizer.step()
            
            c_val  = est_moment(stein_val, h_sample_mb).detach()
          
            epoch_loss += loss.item()/sample.size(0)

        # track loss each epoch
        losses.append(epoch_loss)    

        grad_norms.append(get_grad_norm(net))
        est_moments.append(est_moment(stein_val, h_sample_mb).detach().item())

        if resample_:
            sample = target_dist.generate_points(sample.size(0), sample_range).to(device).requires_grad_(True)
            h_sample = h(sample).detach().to(device)
            logp_sample = target_dist.log_prob(sample)
            score_sample = get_grad(logp_sample.sum(), sample).detach()
            
        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Est moment (Stein-val - h): {est_moment(stein_val, h_sample_mb)}, MC est: {h_sample.mean()}, Grad Norm: {grad_norms[-1]}')

    return net #, losses, grad_norms, est_moments


# train with var grad:
# train with var min objective - ie. pass gradient through mean as well
# no c this time 
def train_network_var_min_precomputed_score(net, optimizer, sample, given_score, target_dist, h, epochs, reg = 0., verbose=True, mb_size = None, with_perturb_samples = False):

    # if want to compare with same sample set as neural Stein
    if with_perturb_samples:
        samples_bar = (sample + torch.randn_like(sample)).to(device)
        sample = torch.cat([sample, samples_bar], dim = 0)

    losses = []
    grad_norms = []
    est_moments = []

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = given_score
    #score_sample = get_grad(logp_sample.sum(), sample).detach()
    
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

            stein_val = stein_g_precomputed_score(sample_mb, net, score_sample)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape)
            assert(stein_val.device == h_sample_mb.device)

            # need to be differentiable wrt g here in mean as well
            # this is the key difference from the "train g" then "train c" alternatingly method above
            est_mean = (stein_val - h_sample_mb).mean()

            # empirical variance (note: NOT unbiased variance estimator)
            loss = torch.mean((stein_val - h_sample_mb - est_mean)**2)

            if reg > 0.:
                loss += reg * (net(sample_mb)**2).mean()

            loss.backward()
            optimizer.step()
            
            c_val  = est_moment(stein_val, h_sample_mb).detach()
          
            epoch_loss += loss.item()/sample.size(0)

        # track loss each epoch
        losses.append(epoch_loss)    

        grad_norms.append(get_grad_norm(net))
        est_moments.append(est_moment(stein_val, h_sample_mb).detach().item())

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Est moment (Stein-val - h): {est_moment(stein_val, h_sample_mb)}, MC est: {h_sample.mean()}, Grad Norm: {grad_norms[-1]}')

    return net #, losses, grad_norms, est_moments



# train with var grad: and importance sampling (assumes target_dist density is normalized)
# pass grad thru mean as well
# must pass sample_dist as well (which off-samples are actually obtained from)
def train_network_var_min_IS(net, optimizer, sample, target_dist, sample_dist_log_prob, h, epochs, reg = 0., verbose=True, mb_size = None, with_perturb_samples = False):

    # if want to compare with same sample set as neural Stein
    if with_perturb_samples:
        samples_bar = (sample + torch.randn_like(sample)).to(device)
        sample = torch.cat([sample, samples_bar], dim = 0)

    losses = []
    grad_norms = []
    est_moments = []


    logp_sample = target_dist.log_prob(sample)
    score_sample = get_grad(logp_sample.sum(), sample).detach()

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    # IS h_sample
    #h_sample_IS = h_sample * torch.exp(logp_sample - sample_dist_log_prob(sample)).detach()
    weight_IS = torch.exp(logp_sample - sample_dist_log_prob(sample)).detach()
    
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

            stein_val = stein_g_precomputed_score(sample_mb, net, score_sample)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape)
            assert(stein_val.device == h_sample_mb.device)

            # need to be differentiable wrt g here in mean as well
            # this is the key difference from the "train g" then "train c" alternatingly method above
            est_mean = (weight_IS*(stein_val - h_sample_mb)).mean()

            # empirical variance (note: NOT unbiased variance estimator)
            loss = torch.mean((stein_val - h_sample_mb - est_mean)**2)

            if reg > 0.:
                loss += reg * (net(sample_mb)**2).mean()

            loss.backward()
            optimizer.step()
          
            epoch_loss += loss.item()/sample.size(0)

        # track loss each epoch
        losses.append(epoch_loss)    

        grad_norms.append(get_grad_norm(net))
        est_moments.append(est_moment(stein_val, h_sample_mb).detach().item())



        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Est moment (Stein-val - h): {est_moment(stein_val, h_sample_mb)}, MC est: {h_sample.mean()}, Grad Norm: {grad_norms[-1]}')

    return net #, losses, grad_norms, est_moments



def evaluate_ncv_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000, reg = 0., 
                             given_sample = None, given_score = None, return_learned = False, mb_size = None, resample_ = False):
    # Initialize distribution and MLP network
    net = MLP(n_dims=net_dims, n_out=net_dims)
    net = torch.nn.DataParallel(net)
    net = net.to(device)

    c = torch.tensor([0.0], device = device, requires_grad=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optim_c = torch.optim.Adam([c], lr=1e-3)

    if given_sample is None:
        #   Generate and prepare sample data
        sample = dist.generate_points(n_samples, sample_range).to(device)
        sample.requires_grad = True
    else:
        # copy given samples, and set requires_grad to True
        sample = given_sample.clone().detach().to(device).requires_grad_(True)
        assert(resample_ == False), "Resampling is not possible with given samples!"

    if given_score is None:
        # Train the network and estimate the moment
        trained_net, trained_c = train_network_ncv_loss(net,c, optimizer, optim_c, sample, dist, h, epochs, reg=reg, verbose=True, mb_size = mb_size, resample_ = resample_, sample_range = sample_range)
    else:
        trained_net, trained_c = train_network_ncv_loss_precomputed_score(net,c, optimizer, optim_c, sample, given_score, h, epochs, reg=reg, verbose=True, mb_size = mb_size)

    h_sample = h(sample).detach() 

    if given_score is None:
        est_moment = (h_sample - stein_g(sample, trained_net, dist.log_prob).to(h_sample.device).detach()).mean()
    else:
        est_moment = (h_sample - stein_g_precomputed_score(sample, trained_net, given_score).to(h_sample.device).detach()).mean()
    
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    print(f"Est moment NCV: {est_moment}, trained_c val: {trained_c.item()}")

    if return_learned:
        return est_moment.mean().item(), trained_net, trained_c 
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())


def evaluate_varg_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000, reg = 0., given_sample = None, given_score = None, 
                              return_learned = False, mb_size = None, perturb_samples = False, resample_ = False):
    # Initialize distribution and MLP network
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
        assert(resample_ == False), "Resampling is not possible with given samples!"

    if given_score is None:
        # Train the network and estimate the moment
        trained_net = train_network_var_min(net, optimizer, sample, dist, h, epochs, reg=reg, verbose=True, mb_size = mb_size, 
                                            with_perturb_samples = perturb_samples, resample_ = resample_, sample_range = sample_range)
    else: 
        trained_net = train_network_var_min_precomputed_score(net, optimizer, sample, given_score, dist, h, epochs, reg=reg, verbose=True, mb_size = mb_size, with_perturb_samples = perturb_samples)
        print("\n\nPassing precomputed score to vargrad is not implemented yet!!")
        return exit()

    h_sample = h(sample).detach() 

    if given_score is None:
        est_moment = (h_sample - stein_g(sample, trained_net, dist.log_prob).to(h_sample.device).detach()).mean()
    else:
        est_moment = (h_sample - stein_g_precomputed_score(sample, trained_net, given_score).to(h_sample.device).detach()).mean()
    
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    print(f"Est moment Var Min: {est_moment}")

    if return_learned:
        return est_moment.mean().item(), trained_net
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())



def evaluate_varg_IS_expectation(dist, net_dims, sample_range, n_samples, h, epochs=1000, reg = 0., given_sample = None, given_score = None, return_learned = False, mb_size = None, perturb_samples = False):
    # Initialize distribution and MLP network
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

    # log prob of uniform dist on sample range
    def sample_dist_log_prob(sample):
        dim = sample.shape[-1]
        val = -dim * np.log(sample_range[1] - sample_range[0])
        return val * torch.ones_like(sample)

    if given_score is None:
        # Train the network and estimate the moment
        trained_net = train_network_var_min_IS(net, optimizer, sample, dist, sample_dist_log_prob, h, epochs, reg=reg, verbose=True, mb_size = mb_size, with_perturb_samples = perturb_samples)
    else:
        print("\n\nPassing precomputed score to vargrad is not implemented yet!!")
        raise NotImplementedError
        #return exit()

    h_sample = h(sample).detach() 
    weight_IS = torch.exp(dist.log_prob(sample) - sample_dist_log_prob(sample)).detach()

    if given_score is None:
        est_moment = (weight_IS*(h_sample - stein_g(sample, trained_net, dist.log_prob).to(h_sample.device).detach())).mean()
    else:
        est_moment = (weight_IS*(h_sample - stein_g_precomputed_score(sample, trained_net, given_score).to(h_sample.device).detach())).mean()
    
    #print(f"Estimated moment for E[x**2] with {dist.__class__.__name__}: {abs(dist.second_moment() - est_moment.mean().item())}")
    print(f"Est moment Var Min: {est_moment}")

    if return_learned:
        return est_moment.mean().item(), trained_net
    return est_moment.mean().item() #-abs(est_moment.mean().item() - dist.second_moment())
