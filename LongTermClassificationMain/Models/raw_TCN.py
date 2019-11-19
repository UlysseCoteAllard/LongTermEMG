"""
Implementation taken from: https://github.com/locuslab/TCN and modified to work with 2D inputs
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from LongTermClassificationMain.Models.model_utils import ReversalGradientLayerF

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x = x[:, :, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TemporalBlock, self).__init__()
        # Dilatation=(1, x) because the temporal information is on the second axis
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=(1, dilation)))
        self.chomp1 = Chomp2d(padding[1])
        self.batch_norm1 = nn.BatchNorm2d(num_features=n_outputs, momentum=0.99, eps=1e-3)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout1 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.batch_norm1, self.relu1, self.dropout1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.init_weights()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x, target=True):
        out = self.net(x)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_kernels, number_of_class=11, input_channel=1, kernel_size=(3, 5), dropout=0.5):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_kernels)

        self._kernel_sizes = kernel_size
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else num_kernels[i-1]
            out_channels = num_kernels[i]

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(0, (kernel_size[1]-1) * dilation_size), dropout=dropout))

        self._network_TCN = nn.ModuleList(layers)
        self._output = nn.Linear(num_kernels[-1], number_of_class)
        self._output_discriminator = nn.Linear(num_kernels[-1], 2)  # Two domains: Source and Target
        self._output_shuffled = nn.Linear(num_kernels[-1], 5)  # 5 Possible shuffle (including no shuffle)
        self._output_time_swapped = nn.Linear(num_kernels[-1], 2)  # 2 Possible swap (including no swap)
        print(self)
        print("Number Parameters: ", self.get_n_params())

    def stop_gradient_except_for_batch_norm(self):
        modules = self._modules
        for key in modules:
            if key == "_network_TCN":
                print(modules[key])
                for temporal_block in modules[key]:
                    temporal_block_dict = temporal_block._modules
                    for layer_key in temporal_block_dict:
                        if isinstance(temporal_block_dict[layer_key], nn.Sequential):
                            for layer_sequence in temporal_block_dict[layer_key]:
                                if not isinstance(layer_sequence, nn.BatchNorm2d):
                                    for param in layer_sequence.parameters():
                                        param.requires_grad = False
                        elif not isinstance(temporal_block_dict[layer_key], nn.BatchNorm2d):
                            for param in temporal_block_dict[layer_key].parameters():
                                param.requires_grad = False
            else:
                for param in modules[key].parameters():
                    param.requires_grad = False

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, get_features=False, get_all_tasks_output=False, lambda_value=1.):
        for i, layer in enumerate(self._network_TCN):
            x = layer(x)

        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        features_extracted = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        output = self._output(features_extracted)
        if get_features:
            return features_extracted, output
        elif get_all_tasks_output:
            reversed_layer = ReversalGradientLayerF.grad_reverse(features_extracted, lambda_value)
            output_domain = self._output_discriminator(reversed_layer)

            output_shuffle = self._output_shuffled(features_extracted)

            output_time_swapped = self._output_time_swapped(features_extracted)

            return output, output_domain, output_shuffle, output_time_swapped
        else:
            return output
