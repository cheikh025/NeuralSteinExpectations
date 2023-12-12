import numpy as np
import pymc3 as pm
import time

class LangevinMCMC:
    def __init__(self, target_dist, target_function, dim, step_size):
        self.target_dist = target_dist
        self.h = target_function # h
        self.dim = dim
        self.step_size = step_size 

    def get_distribution(self, target_dist, dim, model):
        with model:
            if target_dist['name'] == 'gaussian':
                x = pm.MvNormal('x', mu=target_dist['mean'], 
                                cov=target_dist['cov'], shape=dim)
            elif target_dist['name'] == 'student':
                x = pm.StudentT('x', nu=target_dist['nu'], shape=dim)
            elif target_dist['name'] == 'exponential':
                x = pm.Exponential('x', lam=target_dist['rate'], shape=dim)
            return x
        

    def compute_expectation(self, num_samples):
        with pm.Model() as model:
            # Define the Gaussian distribution
            x = self.get_distribution(self.target_dist, self.dim, model)
            self.time = time.time()
            # Langevin MCMC sampling
            with model:
                if self.step_size is None:
                     self.step_size = pm.Metropolis() 
                trace_metropolis = pm.sample(num_samples, step=self.step_size, cores=1)
                expectation = np.mean([self.h(trace_metropolis.get_values('x')[i]) for i in range(len(trace_metropolis))])
            self.time = time.time() - self.time
        return expectation, self.time




class HamiltonianMCMC:
    def __init__(self, target_dist, target_function, dim, target_accept):
        self.target_dist = target_dist
        self.dim = dim
        self.target_accept = target_accept 
        self.h = target_function

    def get_distribution(self, target_dist, dim, model):
        with model:
            if target_dist['name'] == 'gaussian':
                x = pm.MvNormal('x', mu=target_dist['mean'], 
                                cov=target_dist['cov'], shape=dim)
            elif target_dist['name'] == 'student':
                x = pm.StudentT('x', nu=target_dist['nu'], shape=dim)
            elif target_dist['name'] == 'exponential':
                x = pm.Exponential('x', lam=target_dist['rate'], shape=dim)
            return x
        
    
    def compute_expectation(self, num_samples):
        with pm.Model() as model:
            # Define the Gaussian distribution
            x = self.get_distribution(self.target_dist, self.dim, model)
            self.time = time.time()
            # Hamiltonian MCMC sampling
            with model:
                trace_hmc = pm.sample(num_samples, target_accept=self.target_accept, cores=1)  # The NUTS sampler is used by default
                expectation = np.mean([self.h(trace_hmc.get_values('x')[i]) for i in range(len(trace_hmc))])
            self.time = time.time() - self.time
        return expectation, self.time


### Test examples

# Parameters
dim = 2
mean = 10 * np.ones(dim)
cov = 3 * np.eye(dim)
target_dist = {'name': 'gaussian', 'mean': mean, 'cov': cov}
target_function = lambda x: np.sum(x ** 2)
num_samples = 100
step_size = None
target_accept = 0.9

# Run the MCMC methods
langevin = LangevinMCMC(target_dist, target_function, dim, step_size)
expectation, L_time = langevin.compute_expectation(num_samples)
print("Estimated expectation with Langevin method:", expectation)
print("Time taken for Langevin method:", L_time)
hamiltonian = HamiltonianMCMC(target_dist, target_function, dim, target_accept)
expectation, H_time = hamiltonian.compute_expectation(num_samples)
print("Estimated expectation with Hamiltonian method:", expectation)
print("Time taken for Hamiltonian method:", H_time)


