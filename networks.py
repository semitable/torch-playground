from torch import nn
from torch.functional import F


class FCNetwork(nn.Module):
    def __init__(self, dims):
        """
        Creates a network using ReLUs between layers and no activation at the end
        :param dims: tuple in the form of (100, 100, ..., 5). for dim sizes
        """
        super().__init__()
        h_sizes = dims[:-1]
        out_size = dims[-1]

        # Hidden layers
        self.hidden = []
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
            self.add_module("hidden_layer" + str(k), self.hidden[-1])

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    @staticmethod
    def calc_layer_size(size, extra):
        if type(size) is int:
            return size
        return extra['size']

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        output = self.out(x)
        return output

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
