import torch
import torch.nn as nn

from diffusers import Mel
class RBM(nn.Module):

    def __init__(self, n_hidden, n_visible, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.n_hidden = n_hidden
        self.n_visible = n_visible

        # interaction energy and external fields
        self.J = nn.Parameter(torch.rand((self.n_hidden, self.n_visible)))
        self.alpha = nn.Parameter(torch.rand(self.n_visible))
        self.beta = nn.Parameter(torch.rand(n_hidden))

        self.h_dist = nn.Parameter(torch.rand(n_hidden))
        self.v_dist = nn.Parameter(torch.rand(n_visible))

    def energy(self, x, h):
        return x.t() @ self.alpha + h.t() @ self.beta + x.t() @ self.J @ h
    def p_h(self, h):
        return nn.functional.sigmoid(self.alpha + self.J.t() @ h)

    def p_x(self, x):
        return nn.functional.sigmoid(self.beta + self.J @ x)

