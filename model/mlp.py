import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_dims, activations=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = []

        if activations is None:
            self.activations = [nn.ReLU() for _ in range(len(layer_dims) - 1)]
        else:
            self.activations = activations

        for dim_in, dim_out in zip(layer_dims[:-1], layer_dims[1:], ):
            self.layers.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for L, f in zip(self.layers[:-1], self.activations):
            x = f(L(x))

        x = self.layers[-1](x)

        return x
