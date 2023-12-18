import numpy as np
import torch 
from torch import cdist
import torch.distributions as tdist
import matplotlib.pyplot as plt
from distributions import Mixture

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

# a version of Unadjusted Langevin Algorithm
class LangevinSampler:
    # init samples should be of shape: (num_chains, dim)
    def __init__(self, log_prob, num_chains, num_samples, burn_in, init_samples, alpha=0.5, beta=0, gamma=0.55, num_L_steps= 10,device='cpu'):
        self.device = device
        self.log_prob = log_prob

        # x shape: (num_chains, dim)
        if init_samples is not None:
            assert(init_samples.shape[0] == num_chains) #print error

            self.x = init_samples.to(device).detach().clone()
        
        self.num_samples, self.dim = init_samples.shape
        self.learning_rate = self.learning_rate_decay(alpha, beta, gamma)
        self._mean = torch.zeros(self.dim).to(self.device)
        self._cov_identity = torch.eye(self.dim).to(self.device)

        self.num_chains = num_chains
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.sample_list = []

        #for HMC only 
        self.num_L_steps = num_L_steps
        self.init_lr = alpha

    def gaussian_noise(self, lr):
        noise_sampling = tdist.MultivariateNormal(self._mean, 
                                    covariance_matrix=self._cov_identity).sample(sample_shape=torch.Size((self.num_chains,)))
        return lr**0.5 * noise_sampling

    def learning_rate_decay(self, a, b, c):
        """
        Generator for the step sizes.
        Step size schedule a(b+t)^(-c), where t is the step.
        """
        t = 0
        while (True):
            t += 1
            yield a*(b+t)**(-c)

    def step(self):
        self.x = self.x.to(self.device)
        self.x = self.x.detach().requires_grad_(True)

        lr = next(self.learning_rate)

        noise = self.gaussian_noise(lr)

        log_prob_val = self.log_prob(self.x)
        
        grad_res = torch.autograd.grad(log_prob_val.sum(), self.x)[0]

        #print("grad res shape: {}, noise shape: {}".format(grad_res.shape, noise.shape))

        self.x = self.x + (0.5 * lr) * grad_res + noise

    # not carefully tested
    def mala_step(self):
        self.x = self.x.to(self.device)
        self.x = self.x.detach().requires_grad_(True)

        lr = next(self.learning_rate)

        noise = self.gaussian_noise(lr)

        log_prob = self.log_prob(self.x)
        
        grad_res = torch.autograd.grad(log_prob.sum(), self.x)[0]

        proposal = self.x + (0.5 * lr) * grad_res + noise

        log_prob_proposal = self.log_prob(proposal)
        
        grad_proposal = torch.autograd.grad(log_prob_proposal.sum(), proposal)[0]
        log_accept = log_prob_proposal - log_prob - 0.5 * ((proposal - self.x - 0.5 * lr * grad_res)**2).sum() + 0.5 * ((self.x - proposal - 0.5 * lr * grad_proposal)**2).sum()

        # Accept or reject proposal
        if torch.log(torch.rand(1)) < log_accept:
            self.x = proposal
            # return the samples shape: (num_samples, num_chains, dim)
    
    # not tested either
    def hmc_step(self):
        self.x = self.x.to(self.device)
        self.x = self.x.detach().requires_grad_(True)
        
        # fix the lr to just alpha for HMC
        lr = self.init_lr #next(self.learning_rate)

        # Initialize momentum
        p = torch.randn_like(self.x)


        x_cur = torch.clone(self.x)

        # Compute initial Hamiltonian
        log_prob_val = self.log_prob(x_cur)
        kinetic_energy = 0.5 * p.pow(2).sum(dim=-1)
        hamiltonian = -log_prob_val.reshape(self.num_chains) + kinetic_energy.reshape(self.num_chains)

        # Leapfrog integration
        for _ in range(self.num_L_steps):
            log_prob_val = self.log_prob(x_cur)
            grad_res = torch.autograd.grad(log_prob_val.sum(), x_cur)[0]
            p = p + 0.5 * lr * grad_res  # half step update for momentum
            x_next = x_cur + lr * p  # full step update for position
            log_prob_val = self.log_prob(x_next)
            grad_res = torch.autograd.grad(log_prob_val.sum(), x_next)[0]
            p = p + 0.5 * lr * grad_res  # half step update for momentum
            x_cur = x_next

        # Compute proposed Hamiltonian
        kinetic_energy = 0.5 * p.pow(2).sum(dim=-1)
        proposed_hamiltonian = -log_prob_val.reshape(self.num_chains) + kinetic_energy.reshape(self.num_chains)
        
        # Metropolis-Hastings acceptance step
        accept_bools = torch.log(torch.rand_like(hamiltonian)) < hamiltonian - proposed_hamiltonian
        if torch.any(accept_bools):
            #print("accept")
            self.x = self.x.detach()
            x_next = x_next.detach()
            self.x[accept_bools] = x_next[accept_bools]

    def sample(self, sampler_type = "ula"):
        # sampler type
        if sampler_type == "ula":
            step_fn = self.step 
        elif sampler_type == "mala":        
            step_fn = self.mala_step
        elif sampler_type == "hmc": 
            step_fn = self.hmc_step
        else:
            step_fn = self.step

        # run the burn in
        for i in range(self.burn_in):
            step_fn()
            
        # for each sample, run the step, and add the samples to list    
        for i in range(self.num_samples):
            step_fn()
            
            self.sample_list.append(self.x.detach().cpu().numpy())
        self.sample_list_np = np.array(self.sample_list)
        return self.sample_list_np

    # evaluate expectation along each chain 
    # shape: (num_chains, )    
    def eval_expectation(self, h, return_samples=False):
        print("samples shape: ", self.sample_list_np.shape)
        #print("samples: ", self.sample_list_np)
        
        # sample shape: (num_samples, num_chains, dim)
        if return_samples:
            return np.mean(h(self.sample_list_np), axis=0), self.sample_list_np
        return np.mean(h(self.sample_list_np), axis=0)
    
# to evaluate the expectation of a function h(x) under a distribution dist
def eval_Langevin(dist, dim, h, num_samples=100, num_chains=1, alpha = 1., gamma = 0.2, verbose=False, return_samples = False):
    # to make the initial distribution different from the true distribution
    init_samples = 10 + 10*torch.randn(num_chains, dim).to(device)

    lsampler = LangevinSampler(log_prob=dist.log_prob, 
                num_chains =num_chains, 
                num_samples = num_samples, burn_in= 5000, 
                init_samples=init_samples, 
                alpha= alpha, 
                gamma = gamma)

    # shape of samples: (num_samples, num_chains, dim), np array
    samples = lsampler.sample()

    #print("Expectation from each chain: ", lsampler.eval_expectation(h))
    if verbose:
        print("Expectation from all chains: ", (h(samples)).mean())
    if return_samples:
        return (h(samples)).mean(), samples
    return (h(samples)).mean()

def eval_HMC(dist, dim, h, num_samples=100, num_chains=1, alpha = 5e-2, num_L_steps = 5, verbose= False, return_samples=False):
    # good params for Gaussian are: alpha=  5e-2, num_L_steps=5
    # for mixture try: 
    
    # to make the initial distribution different from the true distribution
    init_samples = 10 + 10*torch.randn(num_chains, dim).to(device)

    lsampler = LangevinSampler(log_prob=dist.log_prob, num_chains =num_chains, 
                num_samples = num_samples, burn_in= 5000, init_samples=init_samples, 
                alpha= alpha, 
                num_L_steps=num_L_steps)

    # shape of samples: (num_samples, num_chains, dim), np array
    samples = lsampler.sample(sampler_type="hmc")
    
    #print("Expectation from each chain: ", lsampler.eval_expectation(h))
    if verbose:
        print("Expectation from all chains: ", (h(samples)).mean())
   
    if return_samples:
        return (h(samples)).mean(), samples
    
    return (h(samples)).mean()

        
def gaussian_grid_2d(size=2, std=.25):
    comps = []
    for i in range(size):
        for j in range(size):
            center = np.array([i, j])
            center = torch.from_numpy(center).float()
            comp = tdist.Normal(center, torch.ones((2,)) * std)
            comps.append(comp)

    pi = torch.ones((size**2,)) / (size**2)
    mog = Mixture(comps, pi)
    return mog


if __name__ == '__main__':
    # mixture of gaussians 
    # dist = gaussian_grid_2d(4)
    # x = mog.sample(1000)

    dist = tdist.MultivariateNormal(torch.Tensor([-0.6871, 0.8010]),
                                                   covariance_matrix=torch.tensor([[0.2260, 0.1652], [0.1652, 0.6779]]))                                        
    x = torch.randn(1000, *dist.event_shape)

    # attempt to visualize
    # plt.scatter(x[:, 0].numpy(), x[:, 1].numpy())
    # plt.savefig("samp.jpg")
    # lp = mog.logprob(x)
    #visualize_flow.visualize_transform(logdensity=mog.logprob)
    # print(lp.size())
    # lp = lp.numpy()
    # plt.hist(lp)
    # plt.show()
    # plt.figure(figsize=(9, 3))
    # visualize_flow.visualize_transform(samples=x, sample_names='MOG', logdensity_names = 'MOG pdf', logdensities=mog.log_prob, npts=800)
    # fig_filename = "fig.jpg"
    # plt.savefig(fig_filename)
    # plt.close()


    sgld = LangevinSampler(dist, x, burn_in=5000, num_samples=1000, num_chains=1, device='cpu')
    
    for i in range(5000):
        sgld.step()
    
    # check to see of mean and var are learned 
    print(torch.mean(sgld.x, dim=0))
    print(torch.var(sgld.x, dim=0))