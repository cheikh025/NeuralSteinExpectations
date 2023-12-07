import torch
import torch.nn as nn
from utils import *
class MLP(nn.Module):
    """Multi-Layer Perceptron class."""
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, dropout=False):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
                layer(n_dims, n_hid),
                nn.SiLU(),
                torch.nn.utils.spectral_norm(layer(n_hid, n_hid)),
                nn.SiLU(),
                torch.nn.utils.spectral_norm(layer(n_hid, n_hid)),
                nn.SiLU(),
                layer(n_hid, n_out)
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out.squeeze()
    
def train_network(net, optimizer, sample, normal_dist, h, epochs):
    for e in range(epochs):
        optimizer.zero_grad()

        # Calculate Stein value
        stein_val = stein_g(sample, net, normal_dist.log_prob)

        # Compute gradients
        grad_s = get_grad(stein_val.sum(), sample)
        grad_h = get_grad(h(sample).sum(), sample)

        # Calculate loss as the squared difference of gradients
        loss = torch.sum((grad_s - grad_h)**2)
        loss.backward()

        # Update network parameters
        optimizer.step()

        if e % 100 == 0:  # Print every 100 epochs
            print(f'Epoch [{e}/{epochs}], Loss: {loss.item()}')
    return net