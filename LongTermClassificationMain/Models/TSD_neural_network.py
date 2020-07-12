import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from LongTermClassificationMain.Models.model_utils import ReversalGradientLayerF


class LinearBlocks(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.5):
        super(LinearBlocks, self).__init__()
        self.fully_connected_1 = weight_norm(nn.Linear(n_inputs, n_outputs))
        self.batch_norm1 = nn.BatchNorm1d(num_features=n_outputs, momentum=0.99, eps=1e-3)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.fully_connected_1, self.batch_norm1, self.relu1, self.dropout1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.init_weights()

    def init_weights(self):
        self.fully_connected_1.weight.data.normal_(0, 0.01)

    def forward(self, x, target=True):
        out = self.net(x)
        return out


class TSD_Network(nn.Module):
    def __init__(self, num_neurons, feature_vector_input_length, number_of_class=11, dropout=0.5):
        super(TSD_Network, self).__init__()
        layers = []
        num_levels = len(num_neurons)

        for i in range(num_levels):
            in_neurons = feature_vector_input_length if i == 0 else num_neurons[i - 1]
            out_neurons = num_neurons[i]
            layers.append(LinearBlocks(in_neurons, out_neurons, dropout=dropout))

        self._network = nn.ModuleList(layers)
        self._output = nn.Linear(num_neurons[-1], number_of_class)
        self._output_discriminator = nn.Linear(num_neurons[-1], 2)  # Two domains: Source and Target
        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def apply_dropout(self, module):
        if type(module) == nn.Dropout2d or type(module) == nn.Dropout:
            module.train()

    def freeze_all_except_BN(self):
        # Freeze the weights of the pre-trained model so they do not change during training of the target network
        # (except for the BN layers that will be trained as normal).
        network_modules = self._network._modules
        for key in network_modules:
            if isinstance(network_modules[key], LinearBlocks):
                linear_modules = network_modules[key]._modules
                for key_convBlock in linear_modules:
                    if not isinstance(linear_modules[key_convBlock], nn.BatchNorm2d):
                        if isinstance(linear_modules[key_convBlock], nn.Sequential):
                            sequential_modules = linear_modules[key_convBlock]._modules
                            for key_sequential in sequential_modules:
                                if not isinstance(sequential_modules[key_sequential], nn.BatchNorm2d):
                                    for param in sequential_modules[key_sequential].parameters():
                                        param.requires_grad = False
                        else:
                            for param in linear_modules[key_convBlock].parameters():
                                param.requires_grad = False
            elif not isinstance(network_modules[key], nn.BatchNorm2d):
                for param in network_modules[key].parameters():
                    param.requires_grad = False

    def forward(self, x, get_features=False, get_all_tasks_output=False, lambda_value=1.):
        for i, layer in enumerate(self._network):
            x = layer(x)

        output = self._output(x)
        if get_features:
            return x, output
        elif get_all_tasks_output:
            reversed_layer = ReversalGradientLayerF.grad_reverse(x, lambda_value)
            output_domain = self._output_discriminator(reversed_layer)

            return output, output_domain
        else:
            return output
