import torch
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import neural_CV
import utils
import network 
import neuralStein 

MEAN = 1.0
STD = 2.0 

def integrand(x):  
    return x 

# target dist
dist = torch.distributions.Normal(MEAN, STD)

# gaussian logprob
def logprob(x):
    return dist.log_prob(x)

# alternative to estimating moment without using c
# I think, that as c is trained to optimality it should converge to this number ! 
# check this!
def est_moment(stein_val, h_sample):
    return ((h_sample - stein_val)).mean()

# for a given value of c, the solution to the differential equation:
# d/dx g(x) + d/dx logprob(x) * g(x) = x - c
# for logprob(x) = -1/2 * (x - MEAN)**2 / STD**2
# is g(x) = 
# this is the general solution ! 
def true_g(x,c,k = 0.):
    return np.sqrt( 2 * np.pi) * (1. - c) * torch.erf((x - 1.)/(np.sqrt(2) * 2.)) * torch.exp((1/8)*(x-1.)**2) - 4 + k*torch.exp((1/4)*(x**2 / 2 - x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train with Neural Stein
def train_neural_stein(net, optimizer, sample, target_dist, h, epochs, verbose = True, mb_size = None, multiparticle = False):


    #perturbed samples
    sample_bar = sample + (torch.randn_like(sample)).to(device)

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    h_sample_bar = h(sample_bar).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = utils.get_grad(logp_sample.sum(), sample).detach()

    logp_sample_bar = target_dist.log_prob(sample_bar)
    score_sample_bar = utils.get_grad(logp_sample_bar.sum(), sample_bar).detach()

    # data minibatches
    # full batch 
    if mb_size is None:
        batch_idx = torch.arange(0,sample.size(0)).long()
        mb_size = sample.size(0) 
    else:
        print("**Minibatches with batch size: ", mb_size)
        batch_idx = torch.randperm(sample.size(0))

    # for logging
    losses = []
    errors_soln_c = []
    grad_norms = []
    est_moments = []

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

            stein_val = neural_CV.stein_g_precomputed_score(sample_mb, net, score_sample_mb)
            stein_val_bar = neural_CV.stein_g_precomputed_score(sample_bar_mb, net, score_sample_bar_mb)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape), f"Stein val shape: {stein_val.shape}, H sample shape: {h_sample_mb.shape}"
            assert(stein_val.device == h_sample_mb.device)

            if multiparticle:
                loss = (torch.mean(stein_val - h_sample_mb) - torch.mean(stein_val_bar - h_sample_bar_mb))**2
            else:
                loss = torch.mean(( (stein_val - h_sample_mb) - (stein_val_bar - h_sample_bar_mb))**2)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()/sample.size(0)
        
        # track loss each epoch
        losses.append(epoch_loss)    
        error_soln_c = ((net(sample).detach() - true_g(sample, 1.))**2).mean().item()
        errors_soln_c.append(error_soln_c)
        grad_norms.append(utils.get_grad_norm(net))
        
        stein_val = neural_CV.stein_g_precomputed_score(sample, net, score_sample)
        est_moment_val = est_moment(stein_val, h_sample)

        est_moments.append(est_moment_val.detach().item())

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Est Moment: {est_moment_val}, Error soln_c (on samples): {error_soln_c}, Grad Norm: {grad_norms[-1]}')
             

    return net, losses, errors_soln_c, grad_norms, est_moments
    

# to train neural CV
def train_network_ncv_loss(net, c, optimizer, optim_c, sample, target_dist, h, epochs, reg = 0., verbose=True, mb_size = None):
    losses = []
    errors_soln_c = []
    grad_norms = []
    est_moments = []

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = utils.get_grad(logp_sample.sum(), sample).detach()
    
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
            c.data = est_moment(neural_CV.stein_g_precomputed_score(sample, net, score_sample), h_sample)

        for b_num in range(0, sample.size(0), mb_size):
            idx = batch_idx[b_num: b_num + mb_size]

            # get minibatch vals
            sample_mb = sample[idx, :]
            h_sample_mb = h_sample[idx]

            optimizer.zero_grad()
            optim_c.zero_grad()

            stein_val = neural_CV.stein_g_precomputed_score(sample_mb, net, score_sample)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')
              
            assert(stein_val.shape == h_sample_mb.shape)
            assert(stein_val.device == h_sample_mb.device)
        
            loss = torch.mean((stein_val + c - h_sample_mb)**2)

            if reg > 0.:
                loss += reg * (net(sample_mb)**2).mean()

            loss.backward()
            
            optimizer.step()

            #print("c grad after first optim: ", c.grad)
            optim_c.step() 
            
            epoch_loss += loss.item()/sample.size(0)

        # track loss each epoch
        losses.append(epoch_loss)    
        error_soln_c = ((net(sample).detach() - true_g(sample, c))**2).mean().item()
        errors_soln_c.append(error_soln_c)
        grad_norms.append(utils.get_grad_norm(net))
        est_moments.append(est_moment(stein_val, h_sample_mb).detach().item())

        if verbose:
            if e % 100 == 0:  
                
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Error soln_c: {error_soln_c}, Grad Norm: {grad_norms[-1]}, c value: {c.item()}, Est moment (Stein-val - h): {est_moment(stein_val, h_sample)}')

    return net, c, losses, errors_soln_c, grad_norms, est_moments

# train with var grad:
# train with var min objective - ie. pass gradient through mean as well
# no c this time 
def train_network_var_min(net, optimizer, sample, target_dist, h, epochs, reg = 0., verbose=True, mb_size = None, with_perturb_samples = False):
    # if want to compare with same sample set as neural Stein
    if with_perturb_samples:
        samples_bar = (sample + torch.randn_like(sample)).to(device)
        sample = torch.cat([sample, samples_bar], dim = 0)

    net = net.to(device)

    losses = []
    errors_soln_c = []
    grad_norms = []
    est_moments = []

    # precompute h(sample) and logp(sample)
    h_sample = h(sample).detach().to(device)

    logp_sample = target_dist.log_prob(sample)
    score_sample = utils.get_grad(logp_sample.sum(), sample).detach()
    
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

            stein_val = neural_CV.stein_g_precomputed_score(sample_mb, net, score_sample)

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

        #error_soln_c = ((net(sample).detach() - true_g(sample, c_val))**2).mean().item()
        error_soln_c = ((net(sample).detach() - true_g(sample, 1.0))**2).mean().item()
        
        errors_soln_c.append(error_soln_c)
        grad_norms.append(utils.get_grad_norm(net))
        est_moments.append(est_moment(stein_val, h_sample_mb).detach().item())

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Error g_c: {error_soln_c}, Est moment (Stein-val - h): {est_moment(stein_val, h_sample_mb)}, MC est: {h_sample.mean()}, Grad Norm: {grad_norms[-1]}')

    return net, losses, errors_soln_c, grad_norms, est_moments


# mb_size = None means no minibatching/full batch 
def train_grad_loss(net, optimizer, sample, target_dist, h, epochs, verbose=True, mb_size = None):

    # precompute h(sample) and logp(sample)
    h_sample = h(sample)

    losses = []
    errors_soln_c = []
    grad_norms = []
    est_moments = []

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

            stein_val = neuralStein.stein_g(sample_mb, net, target_dist.log_prob)

            #print(f'Stein val shape: {stein_val.shape}')
            #print(f'H sample shape: {h_sample.shape}')

            assert(stein_val.shape == h_sample_mb.shape), f"Stein val shape: {stein_val.shape}, H sample shape: {h_sample.shape}"

            grad_s = utils.get_grad(stein_val.sum(), sample_mb)
            grad_h = utils.get_grad(h(sample_mb).sum(), sample_mb) # NOTE: preferably uses precomputed h_sample_mb, look into improving
        
            assert(grad_s.shape == grad_h.shape)

            loss = torch.sum((grad_s - grad_h)**2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()/sample.size(0)

        losses.append(epoch_loss)
        error_soln_c = ((net(sample).detach() - true_g(sample, 1.0))**2).mean().item()
        errors_soln_c.append(error_soln_c)
        grad_norms.append(utils.get_grad_norm(net))
        est_moments.append(est_moment(stein_val, h_sample_mb).detach().item())

        if verbose:
            if e % 100 == 0:  
                print(f'Epoch [{e}/{epochs}], Loss: {epoch_loss}, Error g_c: {error_soln_c}, Est moment (Stein-val - h): {est_moment(stein_val, h_sample_mb)}, Grad Norm: {grad_norms[-1]}')
            
    return net, losses, errors_soln_c, grad_norms, est_moments

# train func
def train(seed = 1, epochs = 1000, gt = False, type = "var_min", multiparticle = False):
    
    #set seeds for repro
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if gt:
        # sample from gaussian
        samples = dist.sample((100,1))
    else:
        # off samples
        samples = 0.5*dist.sample((100,1)) - 1.0
        #samples = torch.linspace(-10,10,100).reshape(100, 1) - 5.0
        #samples2 = samples + torch.randn_like(samples)
        #samples = torch.cat([samples, samples2], dim = 0)
    #samples.requires_grad = True 
    samples = samples.to(device).requires_grad_()

    net = network.MLP(n_dims=1, n_out=1)
    net = torch.nn.DataParallel(net)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    if type == "var_min":
        net, losses, errors_g, grad_norms, est_moments = train_network_var_min(net, 
                           optimizer, sample = samples, 
                           target_dist = dist, 
                           h = integrand, epochs = epochs, 
                           reg = 0., verbose=True, 
                           mb_size = None, 
                           with_perturb_samples=True)
        loss_str = "Var Min Loss"
    elif type == "neural_cv":
        c = torch.tensor([0.0], requires_grad=True, device = device)
        optim_c = torch.optim.Adam([c], lr=5e-2)
        
        # train neural CV
        net, c, losses, errors_g, grad_norms, est_moments = train_network_ncv_loss(net, c, 
                           optimizer, 
                           optim_c, sample = samples, 
                           target_dist = dist, 
                           h = integrand, epochs = epochs, 
                           reg = 0., verbose=True, 
                           mb_size = None)
        loss_str = "NCV Loss"
    elif type == "neural_stein":
        net, losses, errors_g, grad_norms, est_moments = train_neural_stein(net,
                                optimizer, sample = samples, 
                                target_dist = dist, 
                                h = integrand, epochs = epochs, 
                                verbose=True, 
                                mb_size = None,
                                multiparticle = multiparticle)
        loss_str = "NeuralStein Diff Loss"
    elif type == "grad":
        net, losses, errors_g, grad_norms, est_moments = train_grad_loss(net,
                                                                         optimizer, sample = samples, 
                                target_dist = dist, 
                                h = integrand, epochs = epochs, 
                                verbose=True, 
                                mb_size = None)
        loss_str = "Grad Loss"
    if gt:
        gt_str = "Ground Truth"
    else:
        gt_str = "Off Sample"

    plt.figure()
    plt.plot(losses, label  = loss_str)
    plt.legend(loc='best')
    plt.title("{} Training {}".format(gt_str, loss_str))
    plt.show()
    plt.savefig("./plots/nn_loss/loss_{}_{}_loss.png".format(type, gt_str))

    plt.plot(errors_g, label = "Error g (at each c)")
    plt.legend(loc='best')
    plt.title("{} {} Error from Solution to ODE_c".format(gt_str, loss_str))
    plt.show()
    plt.savefig("./plots/nn_loss/error_g_{}_{}_loss.png".format(type, gt_str))

    plt.figure()
    plt.plot(grad_norms, label = "Grad Norms")
    plt.legend(loc='best')
    plt.title("Grad Norms over training")
    plt.show()
    plt.savefig("./plots/nn_loss/grad_norms_{}_{}.png".format(type, gt_str))

    if type == "var_min":
        stein_vals = neural_CV.stein_g(samples.to(device), net.to(device), dist.log_prob).to(samples.device)
        h_vals = integrand(samples)
        c_val = est_moment(stein_vals, h_vals).detach().item()
    elif type == "neural_cv":
        c_val = c.detach().item()   
    elif type == "neural_stein":
        c_val = 1.0

    # plot net vs true g
    x = torch.linspace(-10,10,100).view(-1,1)
    
    plt.figure()
    plt.plot(x.detach().numpy(), net(x.to(device)).detach().cpu(), label = "Net g(x)")
    plt.plot(x.detach().numpy(), true_g(x, c_val).detach().cpu(), label = "True g(x)")
    plt.legend(loc='best')
    plt.title("g(x) vs True g(x) for c = {}".format(c_val))
    plt.show()
    plt.savefig("./plots/nn_loss/g_func_{}_{}.png".format(type, gt_str))


    # plot zoomed in net vs true g
    x_zoom = torch.linspace(-3,3,100).view(-1,1)
    plt.figure()
    plt.plot(x_zoom.detach().numpy(), net(x_zoom.to(device)).detach().cpu(), label = "Net g(x)")
    plt.plot(x_zoom.detach().numpy(), true_g(x_zoom, c_val).detach().cpu(), label = "True g(x)")
    plt.legend(loc='best')
    plt.title("Zoomed g(x) vs True g(x) for c = {}".format(c_val))
    plt.show()
    plt.savefig("./plots/nn_loss/zoomed_g_func_{}_{}.png".format(type, gt_str))


    plt.figure()
    plt.plot(x.detach().numpy(), net(x.to(device)).detach().cpu(), label = "Net g(x)")
    plt.title("Net g(x)")
    plt.show()
    plt.savefig("./plots/nn_loss/net_g_func_{}_{}.png".format(type, gt_str))

    return losses, errors_g, grad_norms, est_moments


def main():
    #loss_grad, errors_g_grad, g_norm_grad, est_moments_grad = train(seed = 123, epochs = 1000, gt = False, type = "grad", multiparticle = False)
    #loss_var, errors_g_var, g_norm_var, est_moments_var = train(seed = 123, epochs = 1000, gt = False, type = "var_min", multiparticle = False)
    #loss_diff, errors_g_diff, g_norm_diff, est_moments_diff = train(seed = 123, epochs = 1000, gt = False, type = "neural_stein", multiparticle = False)
    loss_ncv, errors_g_ncv, g_norm_ncv, est_moments_ncv = train(seed = 123, epochs = 1000, gt = False, type = "neural_cv", multiparticle = False)


    plt.figure()
    plt.plot(loss_var, label = "Var Min Loss")
    plt.plot(loss_diff, label = "Neural Stein Loss")
    plt.plot(loss_ncv, label = "Neural CV Loss")
    plt.plot(loss_grad, label = "Grad Loss")
    plt.legend(loc='best')
    plt.xlabel("Iter")
    plt.ylabel('Loss')
    plt.title("Comparing Losses")
    plt.show()
    plt.savefig("./plots/nn_loss/all_losses_comparison.png")

    # plot log loss for var min and neural stein
    plt.figure()
    plt.plot(np.log(loss_var), label = "Var Min Loss")
    plt.plot(np.log(loss_diff), label = "Neural Stein Loss")
    plt.plot(np.log(loss_ncv), label = "Neural CV Loss")
    plt.plot(np.log(loss_grad), label = "Grad Loss")
    plt.legend(loc='best')
    plt.xlabel("Iter")
    plt.ylabel('Log Loss')
    plt.title("Log Loss")
    plt.show()
    plt.savefig("./plots/nn_loss/all_log_loss.png")

    plt.figure()
    plt.plot(g_norm_var, label = "Var Min Grad Norm")
    plt.plot(g_norm_diff, label = "Neural Stein Grad Norm")
    plt.plot(g_norm_ncv, label = "Neural CV Grad Norm")
    plt.plot(g_norm_grad, label = "Grad Loss Grad Norm")
    plt.legend(loc='best')
    plt.xlabel("Iter")
    plt.ylabel('Grad Norm')
    plt.title("Comparison of Grad Norm")
    plt.show()
    plt.savefig("./plots/nn_loss/all_grad_norm.png")

    # plot log of grad norm for var min and neural stein
    plt.figure()
    plt.plot(np.log(g_norm_var), label = "Var Min Grad Norm")
    plt.plot(np.log(g_norm_diff), label = "Neural Stein Grad Norm")
    plt.plot(np.log(g_norm_ncv), label = "Neural CV Grad Norm")
    plt.plot(np.log(g_norm_grad), label = "Grad Loss Grad Norm")
    plt.legend(loc='best')
    plt.xlabel("Iter")
    plt.ylabel('Log Grad Norm')
    plt.title("Comparison of Log Grad Norms")
    plt.show()
    plt.savefig("./plots/nn_loss/all_log_grad_norm.png")

    # plot estimated moments (true is MEAN)
    plt.figure()
    plt.plot(est_moments_var, label = "Var Min Est Moment")
    plt.plot(est_moments_diff, label = "Neural Stein Est Moment")
    plt.plot(est_moments_ncv, label = "Neural CV Est Moment")
    plt.plot(est_moments_grad, label = "Grad Loss Est Moment")
    plt.plot([MEAN]*len(est_moments_var), label = "True Moment")
    plt.legend(loc='best')
    plt.xlabel("Iter")
    plt.ylabel('Est Moment')
    plt.title("Estimated Moments over Iters")
    plt.show()
    plt.savefig("./plots/nn_loss/all_est_moment.png")

if __name__ == "__main__":
    main()