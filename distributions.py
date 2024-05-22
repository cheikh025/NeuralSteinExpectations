import torch
import scipy.special as sc
import sympy as sp
import math as m
import torch.distributions as tdist


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
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        return self.mean ** 2 + self.std ** 2

class ExponentialDistribution(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def log_prob(self, x):
        return - self.rate * x
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        if sample_range[0] < 0:
            sample_range = (0, sample_range[1])
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        return 2 / self.rate ** 2


class StudentsTDistribution(Distribution):
    def __init__(self, nu):
        self.nu = nu 

    def log_prob(self, x):
        return -0.5 * (self.nu + 1) * torch.log(1 + x**2 / self.nu)
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        if self.nu > 2:
            return self.nu / (self.nu - 2)
        return None
    


class LogisticDistribution(Distribution):
    def __init__(self, mu, s):
        self.mu = mu
        self.s = s

    def log_prob(self, x):
        z = (x - self.mu) / self.s
        return -(z  + 2 * torch.log(1 + torch.exp(-z)))
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        return (self.s * m.pi) ** 2 / 3 + self.mu ** 2
    
    
class KumaraswamyDistribution(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):
        return (self.a - 1) * torch.log(x) + (self.b - 1) * torch.log(1 - x ** self.a)
    
    def generate_points(self, n_samples, sample_range=(0, 1)):
        if sample_range[0] < 0 or sample_range[1] > 1 or sample_range[0] > 1:
            sample_range = (0, 1)
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        return self.b*sc.beta(1 + 2/self.a, self.b)  - (self.b*sc.beta(1 + 1/self.a, self.b))**2 

    
class GammaDistribution(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        return (self.alpha - 1) * torch.log(x) - self.beta * x
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        if self.alpha < 1:
            return self.alpha / self.beta ** 2 
        return self.alpha / self.beta ** 2 - ((self.alpha - 1) / self.beta) ** 2

    
class BetaDistribution(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        return (self.alpha - 1) * torch.log(x) + (self.beta - 1) * torch.log(1 - x)
    
    def generate_points(self, n_samples, sample_range=(0, 1)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        mean = self.alpha / (self.alpha + self.beta)
        var = self.alpha * self.beta / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        return mean ** 2 + var
    
class LevyDistribution(Distribution):
    def __init__(self, mu, c):
        self.mu = mu  # Location parameter
        self.c = c    # Scale parameter

    def log_prob(self, x):
        return -self.c / (2 * (x - self.mu)) -1.5 * torch.log(x - self.mu)
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        #check the range to ensure that the distribution is defined the first range msut be greater than self.mu
        if sample_range[0] < self.mu:
            sample_range = (self.mu, sample_range[1])
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    
class LaplaceDistribution(Distribution):
    def __init__(self, mu, b):
        self.mu = mu  
        self.b = b
    def log_prob(self, x):
        return -torch.abs(x - self.mu) / self.b
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        return 2 * self.b ** 2 + self.mu ** 2
    
class ParetoDistribution(Distribution):
    def __init__(self, alpha, xm):
        self.xm = xm  
        self.alpha = alpha
    def log_prob(self, x):
        return - (self.alpha + 1) * torch.log(x)
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        if sample_range[0] < self.xm:
            sample_range = (self.xm, sample_range[1])
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        if self.alpha > 2:
            mean = self.alpha * self.xm / (self.alpha - 1)
            return self.xm ** 2 * self.alpha / (self.alpha - 1) ** 2 / (self.alpha - 2) + mean ** 2
        return None
    
class WeibullDistribution(Distribution):
    def __init__(self, k, l):
        self.k = k  
        self.l = l
    def log_prob(self, x):
        return (self.k - 1) * torch.log(x) - (x / self.l) ** self.k
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        if sample_range[0] < 0:
            sample_range = (0, sample_range[1])
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        mean = self.l * sc.gamma(1 + 1 / self.k)
        var = self.l ** 2 * (sc.gamma(1 + 2 / self.k) - sc.gamma(1 + 1 / self.k) ** 2)
        return mean ** 2 + var
    
class GumbelDistribution(Distribution):
    def __init__(self, mu, b):
        self.mu = mu  
        self.b = b
    def log_prob(self, x):
        return -(x - self.mu) / self.b - torch.exp(-(x - self.mu) / self.b)
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, 1) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
    def second_moment(self):
        mean = self.mu + self.b * sp.EulerGamma.evalf()
        var = self.b ** 2 * (m.pi ** 2 / 6)
        return mean ** 2 + var



   
class MultivariateNormalDistribution(Distribution):
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        #self.inv_covariance = torch.inverse(covariance)
        #self.det_covariance = torch.det(covariance)
        self.dim = mean.size(0)  
        self.torch_dist = torch.distributions.MultivariateNormal(loc = mean, covariance_matrix = covariance)
    def log_prob(self, x):
        return self.torch_dist.log_prob(x)
        #d = x - self.mean
        #return -0.5 * torch.sum((d @ self.inv_covariance) * d, dim=1) 
    
    def sample(self, n):
        return self.torch_dist.sample(n)

    def generate_points(self, n_samples, sample_range=(-5, 5)):
   
         return torch.rand(n_samples, self.dim) * (sample_range[1] - sample_range[0]) + sample_range[0]

    
class DirichletDistribution(Distribution):
    def __init__(self, alpha):
        self.alpha = alpha  
    def log_prob(self, x):
        return torch.sum((self.alpha - 1) * torch.log(x), dim=1)
    
    def generate_points(self, n_samples, sample_range=(0, 1)):
        random_point = torch.rand(n_samples, self.alpha.size(0))
        return random_point / torch.sum(random_point, dim=1, keepdim=True)
    
class BinghamDistribution(Distribution):
    def __init__(self, C, Z):
        self.C = C  # Orientation matrix
        self.Z = Z  # Diagonal matrix of dispersion parameters

    def log_prob(self, x):
        x_trans = torch.matmul(x, self.C)
        log_prob_unnorm = torch.matmul(x_trans * self.Z, x_trans.t())
        return torch.diag(log_prob_unnorm)
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, self.C.size(0)) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
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
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, self.p) * (sample_range[1] - sample_range[0]) + sample_range[0]

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
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, self.p, self.p) * (sample_range[1] - sample_range[0]) + sample_range[0]

class MultinomialDistribution(Distribution):
    def __init__(self, n, p):
        self.n = n  # Number of trials
        self.p = p  # Probabilities of each outcome

    def log_prob(self, x):
        return torch.sum(x * torch.log(self.p), dim=1)
    
    def generate_points(self, n_samples, sample_range=(0, 1)):
        random_point = torch.rand(n_samples, self.p.size(0))
        return random_point / torch.sum(random_point, dim=1, keepdim=True)*self.n
    
class VonMisesFisherDistribution(Distribution):
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa

    def log_prob(self, x):
        mu = self.mu.view(1, -1)
        dot_product = torch.matmul(mu, x.T).squeeze()
        return self.kappa * dot_product
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        x = torch.randn(n_samples, self.mu.size(0))
        x /= torch.norm(x, dim=1, keepdim=True)
        return x
    
class MultivariateExponentialDistribution(Distribution):
    def __init__(self, rates):
        self.rates = rates

    def log_prob(self, x):
        return -torch.sum(self.rates * x, dim=-1)
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        return torch.rand(n_samples, self.rates.size(0)) * (sample_range[1] - sample_range[0]) + sample_range[0]
    

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
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, self.B.size(0)) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
class MultivariateGammaDistribution(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):

        term1 = (self.a - 1) * torch.log(x)
        term2 = -self.b * x
        return torch.sum(term1 + term2, dim=1)
    
    def generate_points(self, n_samples, sample_range=(0, 5)):
        return torch.rand(n_samples, self.a.size(0)) * (sample_range[1] - sample_range[0]) + sample_range[0]
    
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
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        return torch.rand(n_samples, self.means.size(1)) * (sample_range[1] - sample_range[0]) + sample_range[0]
    

class CustomDistribution(Distribution):
    def __init__(self, score_function, dim):
        self.parameters = None 
        self.score_function = score_function
        self.dim = dim

    def log_prob(self, x):
        #return torch.log(self.score_function(x))
        return self.score_function(x)
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
            return torch.rand(n_samples, self.dim) * (sample_range[1] - sample_range[0]) + sample_range[0]
    def second_moment(self):
        return None

# mixture of gaussian to test
class Mixture:
    def __init__(self, comps, pi):
        self.pi = tdist.OneHotCategorical(probs=pi)
        self.comps = comps
        self.dim = comps[0].mean.shape[-1] #comps[0].dim

    def sample(self, n):
        c = self.pi.sample((n,))
        xs = [comp.sample((n,)).unsqueeze(-1) for comp in self.comps]
        xs = torch.cat(xs, -1)
        x = (c[:, None, :] * xs).sum(-1)
        return x

    def log_prob(self, x):
        lpx = [comp.log_prob(x) for comp in self.comps]
        lpx = [lp.view(lp.size(0), -1).sum(1).unsqueeze(-1) for lp in lpx]
        lpx = torch.cat(lpx, -1) #.clamp(-20, 20)
        logpxc = lpx + torch.log(self.pi.probs[None])
        logpx = logpxc.logsumexp(1)
        return logpx
    
    def generate_points(self, n_samples, sample_range=(-5, 5)):
        # Changed this to return the mesh from the same distribution as MCMC methods
        return 1 + 10*torch.randn(n_samples, self.dim) #torch.rand(n_samples, self.dim) * (sample_range[1] - sample_range[0]) + sample_range[0]

