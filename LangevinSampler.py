import numpy as np
import torch 
from torch import cdist
import torch.distributions as tdist
import matplotlib.pyplot as plt


# device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

# a version of Unadjusted Langevin Algorithm
class LangevinSampler:
    # init samples should be of shape: (num_chains, dim)
    def __init__(self, log_prob, num_chains, num_samples, burn_in, init_samples, alpha=0.5, beta=0, gamma=0.55, device='cpu'):
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

        log_prob = self.log_prob(self.x)
        
        grad_res = torch.autograd.grad(log_prob.sum(), self.x)[0]

        #print("grad res shape: {}, noise shape: {}".format(grad_res.shape, noise.shape))

        self.x = self.x + (0.5 * lr) * grad_res + noise

        
    # return the samples shape: (num_samples, num_chains, dim)
    def sample(self):
        # run the burn in
        for i in range(self.burn_in):
            self.step()
        # for each sample, run the step, and add the samples to list    
        for i in range(self.num_samples):
            self.step()
            self.sample_list.append(self.x.detach().cpu().numpy())
        self.sample_list_np = np.array(self.sample_list)
        return self.sample_list_np

    # evaluate expectation along each chain 
    # shape: (num_chains, )    
    def eval_expectation(self, h):
        print("samples shape: ", self.sample_list_np.shape)
        #print("samples: ", self.sample_list_np)
        
        # sample shape: (num_samples, num_chains, dim)
        return np.mean(h(self.sample_list_np), axis=0)

# mixture of gaussian to test
class Mixture:
    def __init__(self, comps, pi):
        self.pi = tdist.OneHotCategorical(probs=pi)
        self.comps = comps

    def sample(self, n):
        c = self.pi.sample((n,))
        xs = [comp.sample((n,)).unsqueeze(-1) for comp in self.comps]
        xs = torch.cat(xs, -1)
        x = (c[:, None, :] * xs).sum(-1)
        return x

    def log_prob(self, x):
        lpx = [comp.log_prob(x) for comp in self.comps]
        lpx = [lp.view(lp.size(0), -1).sum(1).unsqueeze(-1) for lp in lpx]
        lpx = torch.cat(lpx, -1).clamp(-20, 20)
        logpxc = lpx + torch.log(self.pi.probs[None])
        logpx = logpxc.logsumexp(1)
        return logpx
        
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