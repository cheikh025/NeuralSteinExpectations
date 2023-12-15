import torch
import hamiltorch
import time


class HamiltonianMCMC:
    def __init__(self, log_prob, target_function, dim, step_size,
                 sampler='hmc',
                 num_chains=10):
        self.sampler = sampler
        self.log_prob = lambda v: log_prob(v.unsqueeze(0)) # log probability function
        self.h = target_function  # h
        self.step_size = step_size 
        self.chains = num_chains
        self.params_init = torch.zeros(dim)

    def sample(self, num_samples, burn=1000):
        if self.sampler == 'hmc':
            params_hmc = hamiltorch.sample(log_prob_func=self.log_prob, 
                                params_init=self.params_init, 
                                num_samples=num_samples,
                                step_size=self.step_size,
                                burn=burn, 
                                num_steps_per_sample=self.chains)
        elif self.sampler == 'nuts':
            params_hmc = hamiltorch.sample(log_prob_func=self.log_prob, 
                                params_init=self.params_init,
                                num_samples=num_samples,
                                step_size=self.step_size,
                                num_steps_per_sample=self.chains,
                                sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
                                desired_accept_rate=0.8)
        elif self.sampler == 'rmhmc':
            params_hmc = hamiltorch.sample(log_prob_func=self.log_prob, 
                                params_init=self.params_init, 
                                num_samples=num_samples,
                                step_size=self.step_size,num_steps_per_sample=self.chains, 
                                sampler=hamiltorch.Sampler.RMHMC,
                                integrator=hamiltorch.Integrator.IMPLICIT, 
                                fixed_point_max_iterations=1000,
                                fixed_point_threshold=1e-05)
        return params_hmc
        

    def compute_expectation(self, num_samples):
        # Define the Gaussian distribution
        self.time = time.time()
        samples = self.sample(num_samples, burn=min(num_samples/2, 1000))
        expectation = torch.mean(torch.tensor([self.h(samples[i])
                                for i in range(len(samples))]))
        self.time = time.time() - self.time
        return expectation.item(), self.time
