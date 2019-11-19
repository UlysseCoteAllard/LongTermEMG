import torch
from torch import nn
from torch.autograd import Function
from torch.nn.parameter import Parameter


class WeightLayer(nn.Module):
    def __init__(self, init_value=1):
        super().__init__()
        self.weight = Parameter(torch.tensor(init_value))

    def forward(self, input):
        return input*self.weight


class ScaleLayer(nn.Module):
    def __init__(self, parameters_dimensions=(1, 1, 1, 1), init_value=1):
        super().__init__()
        self.scale = Parameter(torch.ones(parameters_dimensions)*init_value)

    def forward(self, input):
        return input*self.scale


class ReversalLayerF(Function):
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
        return ReversalLayerF.apply(x, constant)
