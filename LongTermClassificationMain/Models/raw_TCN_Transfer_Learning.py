import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from LongTermClassificationMain.Models.raw_TCN import TemporalBlock
from LongTermClassificationMain.Models.model_utils import ReversalGradientLayerF, ScaleLayer


class SourceNetwork(nn.Module):
    def __init__(self, num_kernels, number_of_class=11, input_channel=1, kernel_size=(3, 5),
                 dropout=0.2):
        super(SourceNetwork, self).__init__()
        layers = []
        num_levels = len(num_kernels)

        self._kernel_sizes = kernel_size
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else num_kernels[i - 1]
            out_channels = num_kernels[i]

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(0, (kernel_size[1] - 1) * dilation_size), dropout=dropout))

        self._network_TCN = nn.ModuleList(layers)
        self._output = nn.Linear(num_kernels[-1], number_of_class)

        self._output_discriminator = nn.Linear(num_kernels[-1], 2)  # Two domains: Source and Target

        self._output_discriminator = nn.Linear(num_kernels[-1], 2)  # Two domains: Source and Target

        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, get_all_tasks_output=False, lambda_value=1.):
        for i, layer in enumerate(self._network_TCN):
            x = layer(x)

        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        features_extracted = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        output = self._output(features_extracted)

        if get_all_tasks_output:
            reversed_layer = ReversalGradientLayerF.grad_reverse(features_extracted, lambda_value)
            output_domain = self._output_discriminator(reversed_layer)
            return output, output_domain
        else:
            return output


class TargetNetwork(nn.Module):
    def __init__(self, weight_pre_trained_convNet, num_kernels, number_of_class=11, input_channel=1,
                 kernel_size=(4, 10), dropout=0.5, size_image=(10, 150)):
        super(TargetNetwork, self).__init__()

        layers = []
        num_levels = len(num_kernels)

        self._kernel_sizes = kernel_size
        scale_layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else num_kernels[i - 1]
            out_channels = num_kernels[i]

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(0, (kernel_size[1] - 1) * dilation_size), dropout=dropout))
            scale_layers.append(
                ScaleLayer(parameters_dimensions=(1, 1, size_image[0] - ((self._kernel_sizes[0] - 1) * (i + 1)),
                                                  150)))
        self._scale_layers = nn.ModuleList(scale_layers)

        self._network_TCN_target = nn.ModuleList(layers)
        self._output_target = nn.Linear(num_kernels[-1], number_of_class)

        # Start with the pre-trained model
        # Change to seven for the new gesture target network (number of class)
        pre_trained_model = SourceNetwork(number_of_class=number_of_class, dropout=dropout, num_kernels=num_kernels,
                                          kernel_size=kernel_size)
        self._added_source_network_to_graph = nn.Sequential(*list(pre_trained_model.children()))

        print("Number Parameters: ", self.get_n_params())
        # Load the pre-trained model weights (Source Network)
        pre_trained_model.load_state_dict(weight_pre_trained_convNet, strict=False)

        # Freeze the weights of the pre-trained model so they do not change during training of the target network
        # (except for the BN layers that will be trained as normal).
        self._source_network = pre_trained_model._modules
        for key in self._source_network:
            if key == "_network_TCN":
                print(self._source_network[key])
                for temporal_block in self._source_network[key]:
                    temporal_block_dict = temporal_block._modules
                    for layer_key in temporal_block_dict:
                        print(temporal_block_dict[layer_key])
                        if isinstance(temporal_block_dict[layer_key], nn.Sequential):
                            for layer_sequence in temporal_block_dict[layer_key]:
                                if not isinstance(layer_sequence, nn.BatchNorm2d):
                                    for param in layer_sequence.parameters():
                                        param.requires_grad = False
                        elif not isinstance(temporal_block_dict[layer_key], nn.BatchNorm2d):
                            for param in temporal_block_dict[layer_key].parameters():
                                param.requires_grad = False
            else:
                for param in self._source_network[key].parameters():
                    param.requires_grad = False

        print("KEYS : ", self._source_network.keys())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x, get_features=False, get_all_tasks_output=False, lambda_value=1., max_scale_value=2.):
        for i, (layer_source, layer_target) in enumerate(zip(self._source_network['_network_TCN'],
                                                             self._network_TCN_target)):
            self._scale_layers[i].scale.data = torch.clamp(self._scale_layers[i].scale.data, min=0.,
                                                           max=max_scale_value)

            x_source = layer_source(x, target=False)
            x = layer_target(x) + self._scale_layers[i](x_source)
        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        features_extracted = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        output = self._output_target(features_extracted)
        if get_features:
            return features_extracted, output
        elif get_all_tasks_output:
            reversed_layer = ReversalGradientLayerF.grad_reverse(features_extracted, lambda_value)
            output_domain = self._output_discriminator(reversed_layer)

            return output, output_domain
        else:
            return output
