from torch import nn


class FCNetwork(nn.Module):
    def __init__(self, dims, dropout=False):
        """
        Creates a network using ReLUs between layers and no activation at the end
        :param dims: tuple in the form of (100, 100, ..., 5). for dim sizes
        """
        super().__init__()
        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(nn.ReLU())
            if dropout:
                mods.append(nn.Dropout(p=0.1))
            mods.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))

        self.layers = nn.Sequential(*mods)

    @staticmethod
    def calc_layer_size(size, extra):
        if type(size) is int:
            return size
        return extra["size"]

    def forward(self, x):
        # Feedforward
        return self.layers(x)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
