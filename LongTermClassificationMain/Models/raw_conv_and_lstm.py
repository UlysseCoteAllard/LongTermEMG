import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, number_of_blocks, number_of_features_output, number_of_channels=10,
                 filter_size=(1, 26), dropout_rate=0.5):
        super(ConvNet, self).__init__()
        self._number_of_channel_input = number_of_channels
        self._number_of_features_output = number_of_features_output

        self._filter_size = filter_size

        list_blocks = []
        for i in range(number_of_blocks):
            if i == 0:
                list_blocks.append(self.generate_bloc(block_id=i, number_features_input=1,
                                                      number_of_features_output=self._number_of_features_output[i],
                                                      filter_size=filter_size, dropout_rate=dropout_rate))
            else:
                list_blocks.append(self.generate_bloc(block_id=i,
                                                      number_features_input=self._number_of_features_output[i-1],
                                                      number_of_features_output=self._number_of_features_output[i],
                                                      filter_size=filter_size, dropout_rate=dropout_rate))

        self._features_extractor = nn.ModuleList(list_blocks)

        #self._output = nn.Linear(self._number_of_features_output, number_of_class)

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
            ("dropout2D_" + str(block_id), nn.Dropout2d(p=dropout_rate))
        ]))

        return block

    def forward(self, x):
        for i, block in enumerate(self._features_extractor):
            #x = F.pad(x, (0, 0, 0, self._filter_size[0] - 1), mode='circular')
            for _, layer in enumerate(block):
                x = layer(x)
        # Perform the average pooling channel wise (i.e. for each channel of the armband), take the average output of
        # the features
        #print(x.shape)
        features_extracted = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        #output = self._output(features_extracted)
        #print(features_extracted.shape)
        return features_extracted


class TemporalDistributedConvNet(nn.Module):
    def __init__(self, number_of_class, number_of_blocks, number_of_features_output, number_of_channels=10,
                 filter_size=(1, 26), dropout_rate=0.5, hidden_dim=25, number_of_layers_lstm=1):
        super(TemporalDistributedConvNet, self).__init__()

        self._hidden_dim = hidden_dim
        self._num_of_layers_lstm = number_of_layers_lstm

        self._convNet = ConvNet(number_of_blocks=number_of_blocks, number_of_channels=number_of_channels,
                                number_of_features_output=number_of_features_output,
                                filter_size=filter_size, dropout_rate=0.5)

        self._lstm = nn.LSTM(number_of_features_output[-1], hidden_size=self._hidden_dim,
                             num_layers=self._num_of_layers_lstm)
        self._batch_norm_output_lstm = nn.BatchNorm1d(self._hidden_dim, eps=1e-4)
        self._dropout_output_lstm = nn.Dropout(dropout_rate)
        self._output = nn.Linear(self._hidden_dim, number_of_class)

        self.hidden = self.init_hidden()
        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x):
        array_output = None
        for i in range(x.shape[2]):
            x_to_use = x[:, :, i, :, :].contiguous()
            output_convNet = self._convNet(x_to_use)
            if array_output is None:
                array_output = output_convNet.unsqueeze(0)
            else:
                array_output = torch.cat((array_output, output_convNet.unsqueeze(0)), 0)
        #print(np.shape(array_output))
        self.hidden = self.init_hidden(batch_size=len(array_output[1]))
        lstm_output, _ = self._lstm(array_output, self.hidden)
        lstm_output_last_element_sequence = lstm_output[2]
        dropout_output_lstm = self._dropout_output_lstm(F.leaky_relu(self._batch_norm_output_lstm(
            lstm_output_last_element_sequence), negative_slope=0.1, inplace=True))
        return self._output(dropout_output_lstm)

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self._num_of_layers_lstm, batch_size, self._hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(self._num_of_layers_lstm, batch_size, self._hidden_dim)).cuda())
