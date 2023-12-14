import torch
from metalogistic import MetaLogistic
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


class StudentsTDistribution(Distribution):
    def __init__(self, nu):
        self.nu = nu 

    def log_prob(self, x):
        return -0.5 * (self.nu + 1) * torch.log(1 + x**2 / self.nu)
    

class CustomDistribution(Distribution):
    def __init__(self, score_function):
        self.parameters = None 
        self.score_function = score_function

    def log_prob(self, x):
        return torch.log(self.score_function(x))


class LogisticDistribution(Distribution):
    def __init__(self, mu, s):
        self.mu = mu
        self.s = s

    def log_prob(self, x):
        z = (x - self.mu) / self.s
        return -(z  + 2 * torch.log(1 + torch.exp(-z)))
    
    
class KumaraswamyDistribution(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):
        return (self.a - 1) * torch.log(x) + (self.b - 1) * torch.log(1 - x ** self.a)
    
class GammaDistribution(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):
        return (self.a - 1) * torch.log(x) - x / self.b 
    
class BetaDistribution(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):
        return (self.a - 1) * torch.log(x) + (self.b - 1) * torch.log(1 - x)
    
class LevyDistribution(Distribution):
    def __init__(self, mu, c):
        self.mu = mu  # Location parameter
        self.c = c    # Scale parameter

    def log_prob(self, x):
        return -self.c / (2 * (x - self.mu)) -1.5 * torch.log(x - self.mu)
    
    
class LaplaceDistribution(Distribution):
    def __init__(self, mu, b):
        self.mu = mu  
        self.b = b
    def log_prob(self, x):
        return -torch.abs(x - self.mu) / self.b
    
class ParetoDistribution(Distribution):
    def __init__(self, xm, alpha):
        self.xm = xm  
        self.alpha = alpha
    def log_prob(self, x):
        return - (self.alpha + 1) * torch.log(x)
    
class WeibullDistribution(Distribution):
    def __init__(self, k, l):
        self.k = k  
        self.l = l
    def log_prob(self, x):
        return (self.k - 1) * torch.log(x) - (x / self.l) ** self.k
    
class GumbelDistribution(Distribution):
    def __init__(self, mu, b):
        self.mu = mu  
        self.b = b
    def log_prob(self, x):
        return -(x - self.mu) / self.b - torch.exp(-(x - self.mu) / self.b)
    



   
class MultivariateNormalDistribution(Distribution):
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        self.inv_covariance = torch.inverse(covariance)
        self.det_covariance = torch.det(covariance)
        k = mean.size(0)  

    def log_prob(self, x):
        d = x - self.mean
        return -0.5 * torch.sum((d @ self.inv_covariance) * d, dim=1) 
    
class DirichletDistribution(Distribution):
    def __init__(self, alpha):
        self.alpha = alpha  
    def log_prob(self, x):
        return torch.sum((self.alpha - 1) * torch.log(x), dim=1)
    
class BinghamDistribution(Distribution):
    def __init__(self, C, Z):
        self.C = C  # Orientation matrix
        self.Z = Z  # Diagonal matrix of dispersion parameters

    def log_prob(self, x):
        x_trans = torch.matmul(x, self.C)
        log_prob_unnorm = torch.matmul(x_trans * self.Z, x_trans.t())
        return torch.diag(log_prob_unnorm)
    
class MultivariateTDistribution(Distribution):
    def __init__(self, mu, Sigma, nu):
        self.mu = mu  # Mean vector
        self.Sigma = Sigma  # Scale matrix
        self.nu = nu  # Degrees of freedom
        self.inv_Sigma = torch.inverse(Sigma)  # Inverse of Sigma
        self.p = mu.shape[0]  # Dimensionality

    def log_prob(self, x):
        diff = x - self.mu
        return -0.5 * (self.nu + self.p) * torch.log(1 + (1 / self.nu) * torch.sum((diff @ self.inv_Sigma) * diff, dim=1) )
    
class WishartDistribution(Distribution):
    def __init__(self, V, n):
        self.V = V  # Scale matrix
        self.n = n  # Degrees of freedom
        self.inv_V = torch.inverse(V)  # Inverse of the scale matrix
        self.p = V.shape[0]  # Dimension of the matrices

    def log_prob(self, X):
        trace_term = -0.5*torch.trace(torch.matmul(self.inv_V, X))
        log_det_term = 0.5*(self.n - self.p - 1) * torch.logdet(X)
        return trace_term + log_det_term

class MultinomialDistribution(Distribution):
    def __init__(self, n, p):
        self.n = n  # Number of trials
        self.p = p  # Probabilities of each outcome

    def log_prob(self, x):
        return torch.sum(x * torch.log(self.p), dim=1)
    
class VonMisesFisherDistribution(Distribution):
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa

    def log_prob(self, x):
        mu = self.mu.view(1, -1)
        dot_product = torch.matmul(mu, x.T).squeeze()
        return self.kappa * dot_product
    
class MultivariateExponentialDistribution(Distribution):
    def __init__(self, rates):
        self.rates = rates

    def log_prob(self, x):
        return -torch.sum(self.rates * x, dim=-1)
    
class MultivariateLaplaceDistribution(Distribution):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b
        
    def log_prob(self, x):
        return -torch.sum(torch.abs(x - self.mu)/self.b, dim=-1) 
    

class GaussianBernoulliRBMDistribution(Distribution):
    def __init__(self, B, b, c):
        self.B = B
        self.b = b
        self.c = c

    def log_prob(self, x, h):
        # x is the vector of visible units, h is the vector of hidden units
        # Ensure h consists of {-1, 1}
        assert torch.all((h == 1) | (h == -1)), "Elements of h must be either -1 or 1"

        xBh = 0.5 * torch.matmul(x, torch.matmul(self.B, h))
        bx = torch.dot(self.b, x)
        ch = torch.dot(self.c, h)
        norm_x = -0.5 * torch.sum(x ** 2)

        return xBh + bx + ch + norm_x
    
class MultivariateGammaDistribution(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):

        term1 = (self.a - 1) * torch.log(x)
        term2 = -self.b * x
        return torch.sum(term1 + term2, dim=1)
    
class GaussianMixtureDistribution(Distribution):
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights

    def log_prob(self, x):
        K = self.means.shape[0]
        log_probs = torch.zeros(K, x.shape[0])

        for k in range(K):
            mean = self.means[k]
            cov = self.covariances[k]
            weight = self.weights[k]

            diff = x - mean
            inv_cov = torch.inverse(cov)
            det_cov = torch.det(cov)

            # Unnormalized log probability for each component
            log_prob_k = -0.5 * torch.sum(diff @ inv_cov * diff, dim=1)
            log_prob_k += -0.5 * torch.log(det_cov)
            log_prob_k += torch.log(weight)

            log_probs[k] = log_prob_k

        return torch.logsumexp(log_probs, dim=0)