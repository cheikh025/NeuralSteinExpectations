import torch
import torch.nn as nn
from utils import *

# Swish Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class normalize(nn.Module):        
  def forward(self, inp):
    mean_inp = inp.mean()
    std_inp = inp.std()

    return (inp - mean_inp.detach())/std_inp.detach()

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
    
"""
MLP Block with normalization, and using the Swish activation function
"""
class normalizedMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, dropout=False):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
                normalize(),
                layer(n_dims, n_hid),
                Swish(), 
                layer(n_hid, n_hid),
                Swish(),
                layer(n_hid, n_hid),
                Swish(),
                layer(n_hid, n_out)
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out.squeeze()