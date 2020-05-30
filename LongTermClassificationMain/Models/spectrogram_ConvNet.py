import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from LongTermClassificationMain.Models.model_utils import ReversalGradientLayerF

class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, padding, dropout=0.5):
        super(ConvBlock, self).__init__()
        # Dilatation=(1, x) because the temporal information is on the second axis
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size, padding=padding))
        self.batch_norm1 = nn.BatchNorm2d(num_features=n_outputs, momentum=0.99, eps=1e-3)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout1 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.dropout1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.init_weights()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x, target=True):
        out = self.net(x)
        return out

class SpectrogramConvNet(nn.Module):
    def __init__(self, num_kernels, kernel_size, number_of_class=11, input_channel=4, dropout=0.5):
        super(SpectrogramConvNet, self).__init__()
        layers = []
        num_levels = len(num_kernels)

        self._kernel_sizes = kernel_size
        for i in range(num_levels):
            in_channels = input_channel if i == 0 else num_kernels[i - 1]
            out_channels = num_kernels[i]
            layers.append(ConvBlock(in_channels, out_channels, kernel_size[i], padding=(0, 0), dropout=dropout))

        self._network_TCN = nn.ModuleList(layers)
        self._output = nn.Linear(num_kernels[-1], number_of_class)
        self._output_discriminator = nn.Linear(num_kernels[-1], 2)  # Two domains: Source and Target
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
        network_modules = self._network_TCN._modules
        for key in network_modules:
            if isinstance(network_modules[key], ConvBlock):
                convblock_modules = network_modules[key]._modules
                for key_convBlock in convblock_modules:
                    if not isinstance(convblock_modules[key_convBlock], nn.BatchNorm2d):
                        if isinstance(convblock_modules[key_convBlock], nn.Sequential):
                            sequential_modules = convblock_modules[key_convBlock]._modules
                            for key_sequential in sequential_modules:
                                if not isinstance(sequential_modules[key_sequential], nn.BatchNorm2d):
                                    for param in sequential_modules[key_sequential].parameters():
                                        param.requires_grad = False
                        else:
                            for param in convblock_modules[key_convBlock].parameters():
                                param.requires_grad = False
            elif not isinstance(network_modules[key], nn.BatchNorm2d):
                for param in network_modules[key].parameters():
                    param.requires_grad = False

    def forward(self, x, get_features=False, get_all_tasks_output=False, lambda_value=1.):
        for i, layer in enumerate(self._network_TCN):
            #print("SHAPE: ", np.shape(x))
            x = layer(x)

        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        #print("SHAPE: ", np.shape(x))
        features_extracted = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        #print("SHAPE: ", np.shape(features_extracted))
        output = self._output(features_extracted)
        if get_features:
            return features_extracted, output
        elif get_all_tasks_output:
            reversed_layer = ReversalGradientLayerF.grad_reverse(features_extracted, lambda_value)
            output_domain = self._output_discriminator(reversed_layer)

            return output, output_domain
        else:
            return output
