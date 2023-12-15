import torch
#custom unnormalsied distribution log

def harmonic_oscillator_distribution_log(x, alpha, beta):
    """ Harmonic Oscillator Inspired Distribution - Log version   p(\mathbf{x}) = \exp\left(-\frac{1}{2} \sum_{i=1}^{d} \alpha_i x_i^2 + \beta \sum_{i=1}^{d-1} x_i x_{i+1}\right)"""
    quadratic_term = -0.5 * torch.sum(alpha * x**2)
    coupling_term = beta * torch.sum(x[:-1] * x[1:])
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