import torch
import math as m

#custom unnormalsied distribution log

def harmonic_oscillator_distribution_log(x, alpha, beta):
    """ Harmonic Oscillator Inspired Distribution - Log version   p(\mathbf{x}) = \exp\left(-\frac{1}{2} \sum_{i=1}^{d} \alpha_i x_i^2 + \beta \sum_{i=1}^{d-1} x_i x_{i+1}\right)"""
    quadratic_term = -0.5 * torch.sum(alpha * x**2, dim=-1)
    coupling_term = beta * torch.sum(x[:,:-1] * x[:,1:], dim=-1)
    return quadratic_term + coupling_term

def rotational_symmetry_distribution_log(x, k, gamma, omega, epsilon=1e-10):
    """ Rotational Symmetry Distribution - Log version p(\mathbf{x}) = \exp\left(-\sum_{i=1}^{d} \frac{|x_i|^{k_i}}{\gamma_i}\right) \cos\left(\sum_{i=1}^{d} \omega_i x_i\right) """
    nonlinear_term = -torch.sum(torch.abs(x)**k / gamma)
    cosine_term = torch.cos(torch.sum(omega * x))
    return nonlinear_term + torch.log(torch.clamp(cosine_term, min=epsilon))

def exponential_sine_hybrid_distribution_log(x, sigma, phi):
    """ Exponential-Sine Hybrid Distribution - Log version p(\mathbf{x}) = \exp\left(-\sum_{i=1}^{d}\frac{x_i^2}{2\sigma_i^2}\right) \left(1 + \sum_{i=1}^{d}\sin(\phi_i x_i)^2\right) """
    gaussian_term = -torch.sum(x**2 / (2 * sigma**2))
    sine_term = torch.sum(torch.sin(phi * x)**2)
    return gaussian_term + torch.log(1 + sine_term)

def fractal_inspired_distribution_log(x, sigma, N, epsilon=1e-10):
    """ Fractal-Inspired Distribution - Log version \exp\left(-\sum_{i=1}^{d}\frac{x_i^2}{2\sigma_i^2}\right) \prod_{i=1}^{d} \left(1 + \sum_{n=1}^{N} \frac{\cos(n \pi x_i)}{2^n}\right)"""
    gaussian_term = -torch.sum(x**2 / (2 * sigma**2))
    cosine_sum = torch.sum(torch.stack([torch.cos(n * torch.pi * x) / 2**n for n in range(1, N+1)]), dim=0)
    return gaussian_term + torch.log(torch.clamp(torch.prod(1 + cosine_sum, dim=0), min=epsilon))

# Double banana looking distribution
def double_banana_log_prob(x):
    x = x.T
    return -(((torch.norm(x, p=2, dim=0) - 2.0) / 0.4) ** 2 - torch.log(torch.exp(-0.5 * ((x[0] - 2.0) / 0.6) ** 2) +
                                                                        torch.exp(-0.5 * ((x[0] + 2.0) / 0.6) ** 2)))

# Sinusoidal looking distribution
def sinusoidal_log_prob(x):
    x = x.T
    val= -(0.5 * ((x[1] - torch.sin(2.0 * m.pi * x[0] / 4.0)) / 0.4) ** 2)
    
    #cutoff after [-4,4] interval
    #val[x[0] > 4] = -1000000.0
    #val[x[0] < -4] = -1000000.0
    return val

# Banana Distribution
def banana_log_prob(x):
    bananaDist = torch.distributions.MultivariateNormal(torch.Tensor([0, 4]),
                                                        covariance_matrix=torch.tensor([[1, 0.5], [0.5, 1]]))
    a = 2
    b = 0.2
    y = torch.zeros(x.size())
    y[:, 0] = x[:, 0] / a
    y[:, 1] = x[:, 1] * a + a * b * (x[:, 0] * x[:, 0] + a * a)
    return bananaDist.log_prob(y)


# Donut Distribution
def donut_log_prob(x):
    radius = 2.6
    sigma2 = 0.033
    r = x.norm(dim=1)
    return -(r - radius)**2 / sigma2

# example of a how to use the above distributions

#def harmonic_oscillator(x):
#    return  harmonic_oscillator_distribution_log(x, alpha=torch.tensor([1, 1]), beta=torch.tensor([1, 1]))
#dist = CustomDistribution(harmonic_oscillator, 2)
#Estimated = evaluate_stein_expectation(dist, 2, (0,1), 100)
#print(f"Estimated moment for {dist.__class__.__name__}: {Estimated}")
#
#def rotational_symmetry(x):
#    return  rotational_symmetry_distribution_log(x, k=torch.tensor([1, 1]), gamma=torch.tensor([1, 1]), omega=torch.tensor([1, 1]))
#dist = CustomDistribution(rotational_symmetry, 2)
#Estimated = evaluate_stein_expectation(dist, 2, (0,1), 100)
#print(f"Estimated moment for {dist.__class__.__name__}: {Estimated}")
#
#def exponential_sine_hybrid(x):
#    return  exponential_sine_hybrid_distribution_log(x, sigma=torch.tensor([1, 1]), phi=torch.tensor([1, 1]))
#dist = CustomDistribution(exponential_sine_hybrid, 2)
#Estimated = evaluate_stein_expectation(dist, 2, (0,1), 100)
#print(f"Estimated moment for {dist.__class__.__name__}: {Estimated}")
#
#def fractal_inspired(x):
#    return  fractal_inspired_distribution_log(x, sigma=torch.tensor([1, 1]), N=2)
#dist = CustomDistribution(fractal_inspired, 2)
#Estimated = evaluate_stein_expectation(dist, 2, (0,1), 100)
#print(f"Estimated moment for {dist.__class__.__name__}: {Estimated}")