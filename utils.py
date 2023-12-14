import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

def stein_g(x, g, logp):
    """Compute the Stein operator of g for a given log probability function logp."""
    x.to(device)
    score = get_grad(logp(x).sum(), x)
    gx = g(x).reshape(x.shape)
    trace_j_critic = exact_jacobian_trace(gx, x)
    score_critic_dot = (gx * score).sum(-1)
    stein_val_batches = score_critic_dot + trace_j_critic
    return stein_val_batches

def train_network(net, optimizer, sample, normal_dist, h, epochs):
    for e in range(epochs):
        optimizer.zero_grad()

        stein_val = stein_g(sample, net, normal_dist.log_prob)

        grad_s = get_grad(stein_val.sum(), sample)
        grad_h = get_grad(h(sample).sum(), sample)

        loss = torch.sum((grad_s - grad_h)**2)
        loss.backward()
        optimizer.step()

        if e % 100 == 0:  
            print(f'Epoch [{e}/{epochs}], Loss: {loss.item()}')
    return net

def expectation_sum_of_squares(mean, covariance):
    variances = torch.diag(covariance)  # Extract the variances (diagonal elements of the covariance matrix)
    mean_squares = mean ** 2  # Square of the mean vector
    return torch.sum(variances + mean_squares).item()