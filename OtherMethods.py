import torch
import pymc3 as pm
import time

class LangevinMCMC:
    def __init__(self, target_dist, target_function, dim, step_size,
                 num_chains=10):
        self.target_dist = target_dist
        self.h = target_function # h
        self.dim = dim
        self.step_size = step_size 
        self.model = pm.Model()
        self.chains = num_chains

    def get_distribution(self, target_dist, dim):
        with self.model:
            if target_dist['name'] == 'gaussian':
                x = pm.MvNormal('x', mu=target_dist['mean'], 
                                cov=target_dist['cov'], shape=dim)
            elif target_dist['name'] == 'student':
                x = pm.StudentT('x', nu=target_dist['nu'], shape=dim)
            elif target_dist['name'] == 'exponential':
                x = pm.Exponential('x', lam=target_dist['rate'], shape=dim)
            return x
        

    def compute_expectation(self, num_samples):
        with self.model:
            # Define the Gaussian distribution
            x = self.get_distribution(self.target_dist, self.dim)
            self.time = time.time()
            # Langevin MCMC sampling
            if self.step_size is None:
                 self.step_size = pm.Metropolis() 
            trace_metropolis = pm.sample(num_samples, step=self.step_size, cores=1, chains=self.chains)
            expectation = torch.mean(torch.tensor([self.h(torch.from_numpy(trace_metropolis.get_values('x')[i]))
                                    for i in range(len(trace_metropolis))]))
            self.time = time.time() - self.time
        return expectation.item(), self.time




class HamiltonianMCMC:
    def __init__(self, target_dist, target_function, dim, target_accept,
                 num_chains=10):
        self.target_dist = target_dist
        self.dim = dim
        self.target_accept = target_accept 
        self.h = target_function
        self.model = pm.Model()
        self.chains = num_chains

    def get_distribution(self, target_dist, dim):
        with self.model:
            if target_dist['name'] == 'gaussian':
                x = pm.MvNormal('x', mu=target_dist['mean'], 
                                cov=target_dist['cov'], shape=dim)
            elif target_dist['name'] == 'student':
                x = pm.StudentT('x', nu=target_dist['nu'], shape=dim)
            elif target_dist['name'] == 'exponential':
                x = pm.Exponential('x', lam=target_dist['rate'], shape=dim)
            return x
        
    
    def compute_expectation(self, num_samples):
        with self.model:
            # Define the Gaussian distribution
            x = self.get_distribution(self.target_dist, self.dim)
            self.time = time.time()
            # Hamiltonian MCMC sampling
            trace_hmc = pm.sample(num_samples, target_accept=self.target_accept, cores=1, chains=self.chains)  # The NUTS sampler is used by default
            expectation = torch.mean(torch.tensor([self.h(torch.from_numpy(trace_hmc.get_values('x')[i])) 
                                    for i in range(len(trace_hmc))]))
            self.time = time.time() - self.time
        return expectation.item(), self.time




