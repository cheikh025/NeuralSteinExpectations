import torch
class Distribution:
    def __init__(self, parameters):
        self.parameters = parameters

    def log_prob(self, x):
        raise NotImplementedError
    
    
class NormalDistribution(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def log_prob(self, x):
        return -(x - self.mean) ** 2 / (2 * self.std ** 2)

class ExponentialDistribution(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def log_prob(self, x):
        return - self.rate * x
    
class NormalDistributionKD(Distribution):
    def __init__(self, mean, covariance):

        self.mean = mean
        self.covariance = covariance
        self.inv_covariance = torch.inverse(covariance)
        self.det_covariance = torch.det(covariance)
        k = mean.size(0)  
        self.normalization_factor = torch.sqrt((2 * torch.pi) ** k * self.det_covariance)

    def log_prob(self, x):
        assert x.ndim == 2 and x.size(1) == self.mean.size(0)
        d = x - self.mean
        return -0.5 * torch.sum((d @ self.inv_covariance) * d, dim=1) - torch.log(self.normalization_factor)
    
    

    