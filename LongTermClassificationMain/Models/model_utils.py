import re
import math
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.loss import _WeightedLoss
import torch.distributions.normal as NormalDistribution


def swish(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    if inputs.is_cuda:
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype).cuda()  # uniform [0,1)
    else:
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 reduce_horizontal_by_one=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2
        self._reduce_horizontal_by_one = reduce_horizontal_by_one

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if self._reduce_horizontal_by_one:
            pad_h -= 1
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class FunkyConvBlock(nn.Module):
    def __init__(self, input_filters, final_output, squeeze_excite_ratio, id_skip, stride,
                 kernel_size, groups_nmbr=1):
        super().__init__()
        self._input_filters = input_filters
        self._final_output = final_output
        self._has_se = (squeeze_excite_ratio is not None) and (0 < squeeze_excite_ratio <= 1)
        self._id_skip = id_skip
        self._stride = stride
        self._kernel_size = kernel_size
        self._groups_nmbr = groups_nmbr

        # Convolutional layer with same output size as input when stride = 1.
        # When stride greater, serve as a dimensionality reduction
        #self._conv1 = Conv2dSamePadding(in_channels=input_filters, out_channels=final_output, kernel_size=kernel_size,
        #                                stride=stride, groups=groups_nmbr, reduce_horizontal_by_one=True)

        self._conv1 = nn.Conv2d(in_channels=input_filters, out_channels=final_output, kernel_size=kernel_size,
                                stride=stride)

        self._squeeze_and_excite = SqueezeAndExciteLayer(input_filters=input_filters, output_filters=final_output,
                                                         squeeze_excite_ratio=squeeze_excite_ratio)

        self._bn = nn.BatchNorm2d(num_features=final_output)

    def forward(self, inputs, drop_connect_rate=None):
        #x = swish(self._conv1(inputs))
        x = F.leaky_relu(self._conv1(inputs), negative_slope=0.1, inplace=True)

        if self._has_se:
            x = self._squeeze_and_excite(x)

        x = self._bn(x)

        if self._id_skip and self._stride == 1 and self._input_filters == self._final_output:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection from ResNet

        return x


class SqueezeAndExciteLayer(nn.Module):
    def __init__(self, input_filters, output_filters, squeeze_excite_ratio):
        super().__init__()
        num_squeezed_channels = max(1, int(input_filters*squeeze_excite_ratio))
        self._se_reduce = Conv2dSamePadding(in_channels=output_filters,
                                            out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels,
                                            out_channels=output_filters, kernel_size=1)

    def forward(self, inputs):
        x_squeezed = F.adaptive_avg_pool2d(inputs, 1)
        x_squeezed = self._se_expand(F.leaky_relu(self._se_reduce(x_squeezed), negative_slope=0.1, inplace=True))
        x = torch.sigmoid(x_squeezed) * inputs
        return x


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2))# and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=(int(options['k'][0]), int(options['k'][1])),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=(int(options['s'][0]), int(options['s'][1]))
        )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings

def generate_examples_by_swapping_time_data(examples, number_of_labels=2):
    labels = torch.randint(low=0, high=number_of_labels, size=(examples.size(0),))
    # Batch X kernel X Channel X Time (we want to change the channel
    examples_time_swapped = torch.empty_like(examples)
    mean_examples = torch.mean(examples, dim=[0, 1, 3])
    std_examples = torch.std(examples, dim=[0, 1, 3])
    normalDist = NormalDistribution.Normal(mean_examples, std_examples)
    random_channel = torch.randint(examples.size(2), (examples.size(0),))

    for i, example in enumerate(examples):
        if labels[i] == 1:
            examples_time_swapped[i, 0, random_channel[i], :] = normalDist.sample((examples.size(3),)
                                                                                  ).transpose_(0, 1)[random_channel[i]]
        else:
            examples_time_swapped[i, :, :, :] = examples[i, :, :, :]

        '''
        for j, kernel in enumerate(example):
            
            for k, channel in enumerate(kernel):
                # Randomize the timeserie
                if labels[i] == 1:
                    idx = torch.randperm(channel.size(0))
                    examples_time_swapped[i, j, k, :] = channel[idx]
                else:
                    examples_time_swapped[i, j, k, :] = channel
        '''
    return examples_time_swapped, labels


def generate_examples_by_swapping_channels(examples, number_of_labels=2):
    '''
    channel_swap = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 5, 2, 7, 4, 9, 6, 1, 8, 3], #[4, 1, 6, 3, 8, 5, 0, 7, 2, 9],
                    [0, 1, 7, 8, 9, 5, 6, 2, 3, 4], [5, 6, 7, 3, 4, 0, 1, 2, 8, 9]]
    '''
    channel_swap = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 5, 2, 7, 4, 9, 6, 1, 8, 3]]#, [0, 1, 7, 3, 4, 5, 6, 2, 8, 9]]#,
                    #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    channel_swap = torch.from_numpy(np.array(channel_swap, dtype=np.int64))
    labels = torch.randint(low=0, high=number_of_labels, size=(examples.size(0),))
    # Batch X kernel X Channel X Time (we want to change the channel
    examples_swapped = torch.empty_like(examples)
    for i, example in enumerate(examples):
        for j, kernel in enumerate(example):
            examples_swapped[i, j, :, :] = kernel[channel_swap[labels[i]]]
    return examples_swapped, labels


def generate_task_examples(input_source, input_target, device='cuda'):
    swapped_channels_examples_source, swapped_channels_labels_source = generate_examples_by_swapping_channels(
        input_source)
    swapped_channels_examples_target, swapped_channels_labels_target = generate_examples_by_swapping_channels(
        input_target)
    swapped_channels_examples = torch.cat((swapped_channels_examples_source, swapped_channels_examples_target), dim=0)
    swapped_channels_labels = torch.cat((swapped_channels_labels_source, swapped_channels_labels_target), dim=0)
    idx = torch.randperm(swapped_channels_labels.nelement())
    swapped_channels_examples_shuffled = swapped_channels_examples[idx]
    swapped_channels_labels_shuffled = swapped_channels_labels[idx]

    time_examples_source, time_labels_source = generate_examples_by_swapping_time_data(
        input_source)
    time_examples_target, time_labels_target = generate_examples_by_swapping_time_data(
        input_target)
    time_examples = torch.cat((time_examples_source, time_examples_target), dim=0)
    time_labels = torch.cat((time_labels_source, time_labels_target), dim=0)
    idx = torch.randperm(time_labels.nelement())
    time_examples_shuffled = time_examples[idx]
    time_labels_shuffled = time_labels[idx]


    '''
    time_examples_source, time_labels_source = generate_examples_by_swapping_time_data(
        swapped_channels_examples_shuffled)

    return time_examples_source.to(device=device), swapped_channels_labels_shuffled.to(device=device), \
           time_labels_source.to(device=device)
    '''

    return swapped_channels_examples_shuffled.to(device=device), swapped_channels_labels_shuffled.to(device=device), \
           time_examples_shuffled.to(device=device), time_labels_shuffled.to(device=device)


# Layer by Yann Dubois (https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/3)
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self._sigma = sigma
        self.noise = torch.zeros(1).cuda()

    def forward(self, x):
        if self.training:
            scaling_factor = torch.std(x.detach()) * self._sigma
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scaling_factor
            x += sampled_noise
        return x


class ConditionalEntropyLoss(_WeightedLoss):
    """
    This criterion combines "log_softmax" and the ConditionalEntropy in a single function

    Args:
    input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
        in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
        in the case of K-dimensional loss.
    """
    def __init__(self, weight=None, size_average=None, ignore_index=-100):
        super(ConditionalEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index

    def forward(self, x):
        dim = x.dim()
        if dim < 2:
            raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

        p = F.softmax(x, dim=1)
        log_p = F.log_softmax(x, dim=1)
        # This is based on equation 6 of the DIR-T paper: https://arxiv.org/abs/1802.08735
        return -(p * log_p).sum(1).mean(0)


'''
Apply the KL divergence as a loss (but first compute the log_softmax for predictions from both models). 
As we are in a machine learning context, minimizing the KL divergence is equivalent to minimizing the cross-entropy 
(do that instead)
'''
class KL_divergence_loss(nn.Module):
    def __init__(self):
        super(KL_divergence_loss, self).__init__()

    def forward(self, predictions, predictions_old_model):
        return self.KL_divergence(predictions_old_model, predictions)

    def KL_divergence(self, predictions_q, predictions_p):
        q = F.softmax(predictions_q, dim=1)
        q_log_q = torch.mean(torch.sum(q * F.log_softmax(predictions_q, dim=1), dim=1))
        q_log_p = torch.mean(torch.sum(q * F.log_softmax(predictions_p, dim=1), dim=1))
        return q_log_q - q_log_p

"""
VATLoss implemented using: https://github.com/ozanciga/dirt-t and
https://github.com/domainadaptation/salad 
author={Schneider, Steffen and Ecker, Alexander S. and Macke, Jakob H. and Bethge, Matthias}
"""
class VATLoss(nn.Module):
    def __init__(self, model, epsilon_radius=1e0, perturbation_for_gradient=1e-1, n_power=1):
        super(VATLoss, self).__init__()
        self._model = model
        self._epsilon_radius = epsilon_radius
        self._n_power = n_power
        self._perturbation_for_gradient = perturbation_for_gradient

    def forward(self, x, predictions):
        vat_loss = self.virtual_adversarial_loss(x, predictions)
        return vat_loss

    def generate_virtual_adversarial_pertubation(self, x, predictions, device='cuda'):
        adversarial_perturbation = torch.randn_like(x, device=device)
        for _ in range(self._n_power):
            adversarial_perturbation = self._perturbation_for_gradient * \
                                       self.get_normalized_vector(adversarial_perturbation).requires_grad_()
            predictions_perturbated = self._model(x + adversarial_perturbation)
            divergence = self.KL_divergence(predictions, predictions_perturbated)
            gradient = torch.autograd.grad(divergence, [adversarial_perturbation])[0]
            adversarial_perturbation = gradient.detach()
        return self._epsilon_radius * self.get_normalized_vector(adversarial_perturbation)

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def KL_divergence(self, predictions_q, predictions_p):
        q = F.softmax(predictions_q, dim=1)
        q_log_q = torch.mean(torch.sum(q * F.log_softmax(predictions_q, dim=1), dim=1))
        q_log_p = torch.mean(torch.sum(q * F.log_softmax(predictions_p, dim=1), dim=1))
        return q_log_q - q_log_p

    def virtual_adversarial_loss(self, x, predictions):
        adversarial_perturbation = self.generate_virtual_adversarial_pertubation(x, predictions)
        predictions_detached = predictions.detach()
        adversarial_output_predictions = self._model(x + adversarial_perturbation)
        vatLoss = self.KL_divergence(predictions_detached, adversarial_output_predictions)
        return vatLoss


"""
EMA
This is taken entirely from: https://github.com/ozanciga/dirt-t
"""
class ExponentialMovingAverage:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]


class ReversalGradientLayerF(Function):
    @staticmethod
    def forward(ctx, input, lambda_hyper_parameter):
        ctx.lambda_hyper_parameter = lambda_hyper_parameter
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_hyper_parameter
        return output, None

    @staticmethod
    def grad_reverse(x, constant):
        return ReversalGradientLayerF.apply(x, constant)


class ScaleLayer(nn.Module):
    def __init__(self, parameters_dimensions=(1, 1, 1, 1), init_value=1.):
        super().__init__()
        self.scale = Parameter(torch.ones(parameters_dimensions)*init_value)

    def forward(self, input):
        return input*self.scale
