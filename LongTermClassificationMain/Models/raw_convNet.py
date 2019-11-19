import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from LongTermClassificationMain.Models.model_utils import FunkyConvBlock, BlockDecoder, Conv2dSamePadding, swish,\
    GaussianNoise


class Model(nn.Module):
    def __init__(self, number_of_class, number_of_blocks, number_of_channels=10, number_of_features_output=64,
                 filter_size=(1, 26), dropout_rate=0.5):
        super(Model, self).__init__()
        self._number_of_channel_input = number_of_channels
        self._number_of_features_output = number_of_features_output

        list_blocks = []
        for i in range(number_of_blocks):
            if i == 0:
                list_blocks.append(self.generate_bloc(block_id=i, number_features_input=1,
                                                      number_of_features_output=self._number_of_features_output,
                                                      filter_size=filter_size, dropout_rate=dropout_rate))
            else:
                list_blocks.append(self.generate_bloc(block_id=i, number_features_input=self._number_of_features_output,
                                                      number_of_features_output=self._number_of_features_output,
                                                      filter_size=filter_size, dropout_rate=dropout_rate))

        self._features_extractor = nn.ModuleList(list_blocks)

        self._output = nn.Linear(self._number_of_features_output, number_of_class)

        self._number_of_blocks = number_of_blocks


        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params


    def generate_bloc(self, block_id, number_features_input=64, number_of_features_output=64, filter_size=(1, 26),
                      dropout_rate=0.5):
        block = nn.Sequential(OrderedDict([
            ("conv2D_" + str(block_id), nn.Conv2d(in_channels=number_features_input, out_channels=
            number_of_features_output, kernel_size=filter_size, stride=1)),
            ("batchNorm_" + str(block_id), nn.BatchNorm2d(num_features=number_of_features_output, momentum=0.99,
                                                          eps=1e-3)),
            ("leakyRelu_" + str(block_id), nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ("dropout2D_" + str(block_id), nn.Dropout2d(p=dropout_rate))#,
            #("GaussianNoise_" + str(block_id), GaussianNoise(sigma=0.1))
        ]))

        return block

    def forward(self, x):
        for i, block in enumerate(self._features_extractor):
            for _, layer in enumerate(block):
                x = layer(x)
        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        features_extracted = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        output = self._output(features_extracted)
        return output
