import torch
from torch.utils.data import DataLoader, TensorDataset
from distributions import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_grad(output, input):
    """Compute the gradient of 'output' with respect to 'input'."""
    return torch.autograd.grad(outputs=output, inputs=input,
                               grad_outputs=torch.ones_like(output),
                               create_graph=True, retain_graph=True)[0]

def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = get_grad(fxi.sum(), x)[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.sum(dim=1)

def generate_parameter_variations():
    parameter_variations= {
        NormalDistribution: [
            {'mean': 0, 'std': 1},
            {'mean': 5, 'std': 2},
            {'mean': 10, 'std': 3},
        ],
        ExponentialDistribution: [
            {'rate': 0.5},
            {'rate': 1.0},
            {'rate': 2.5},
        ],
        StudentsTDistribution: [
            {'nu': 2.5},
            {'nu': 5.0},
            {'nu': 10.0},
        ],
        LogisticDistribution: [
            {'mu': 0, 's': 1},
            {'mu': 5, 's': 2},
            {'mu': 10, 's': 3},
        ],
        KumaraswamyDistribution: [
            {'a': 0.5, 'b': 0.5},
            {'a': 1.0, 'b': 1.0},
            {'a': 2.0, 'b': 2.0},
        ],
        GammaDistribution: [
            {'alpha': 1.0, 'beta': 0.5},
            {'alpha': 2.0, 'beta': 1.0},
            {'alpha': 3.0, 'beta': 1.5},
        ],
        LaplaceDistribution: [
            {'mu': 0, 'b': 1},
            {'mu': 5, 'b': 2},
            {'mu': 10, 'b': 3},
        ],
        BetaDistribution: [
            {'alpha': 0.5, 'beta': 0.5},
            {'alpha': 1.0, 'beta': 1.0},
            {'alpha': 2.0, 'beta': 2.0},
        ],
        ParetoDistribution: [
            {'alpha': 2.5, 'xm': 0.5},
            {'alpha': 10.0, 'xm': 1.0},
            {'alpha': 21.0, 'xm': 2.0},
        ],
        WeibullDistribution: [
            {'k': 0.5, 'l': 0.5},
            {'k': 1.0, 'l': 1.0},
            {'k': 2., 'l': 2.0},
        ],
        GumbelDistribution: [
            {'mu': 0, 'b': 1},
            {'mu': 5, 'b': 2},
            {'mu': 10, 'b': 3},
        ],
    }
    return parameter_variations



def generate_distribution_params(distribution_type, dim):
    if distribution_type == "DirichletDistribution":
        # alpha should be a positive vector
        alpha = torch.rand(dim) + 0.01  # Adding 0.1 to ensure positivity
        return {'alpha': alpha}

    elif distribution_type == "MultivariateTDistribution":
        # Generate mean (mu) and covariance
        mu = torch.rand(dim)
        A = torch.rand(dim, dim)
        covariance = torch.mm(A, A.t())
        nu = torch.randint(1, 10, (1,)).item()  # A random integer
        return {'mu': mu, 'Sigma': covariance, 'nu': nu}

    elif distribution_type == "MultinomialDistribution":
        # 'n' is a scalar, 'p' is a probability vector that sums to 1
        n = torch.randint(1, 10, (1,)).item()  # A random integer
        p = torch.rand(dim)
        p = p / p.sum()  # Normalize to sum to 1
        return {'n': n, 'p': p}

    elif distribution_type == "VonMisesFisherDistribution":
        # 'mu' is a unit vector, 'kappa' is a concentration parameter
        mu = torch.rand(dim)
        mu = mu / torch.norm(mu)  # Normalize to unit length
        kappa = torch.rand(1).item()
        return {'mu': mu, 'kappa': kappa}

    elif distribution_type == "MultivariateNormalDistribution":
        # Generate mean (mu) and covariance
        mu = torch.rand(dim)
        A = torch.rand(dim, dim)
        covariance = torch.mm(A, A.t())
        return {'mean': mu, 'covariance': covariance}

    else:
        raise ValueError("Unsupported distribution type")

def expectation_sum_of_squares_normal(mean, covariance):
    variances = torch.diag(covariance) 
    mean_squares = mean ** 2  
    return torch.sum(variances + mean_squares).item()

def get_grad_norm(net):
    grad_norm = 0.
    for p in net.parameters():
        grad_norm += p.grad.norm(2).item()**2
    return grad_norm**0.5
