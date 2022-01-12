import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group


def deterministic_binarize(tensor):
    return torch.sign(tensor)


def stochastic_binarize(tensor):
    return tensor.add_(1).div_(2).add_(
        torch.rand(tensor.size()).to(tensor.device).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class XNOR_BinarizeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, padding_mode='zeros', binary_func="deter"):
        super(XNOR_BinarizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                  padding, dilation, groups, bias, padding_mode)
        self.binary_func = binary_func

        w = self.weight  # [c_out, c_in, h, w]
        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1).detach()
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)

    def forward(self, input):
        a0 = input
        w = self.weight
        w1 = w - w.mean([1, 2, 3], keepdim=True)
        w2 = w1 / w1.std([1, 2, 3], keepdim=True)
        a1 = a0 - a0.mean([1, 2, 3], keepdim=True)
        a2 = a1 / a1.std([1, 2, 3], keepdim=True)

        bw = XNOR_BinaryQuantize().apply(w2)
        ba = XNOR_BinaryQuantize_a().apply(a2)
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        # import pdb;pdb.set_trace()
        output = output * self.alpha
        return output


class XNOR_BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # import pdb; pdb.set_trace()
        input = ctx.saved_tensors
        grad_input = grad_output.clone().clamp(min=-1, max=1)
        return grad_input


class XNOR_BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        # import pdb;pdb.set_trace()
        ctx.save_for_backward(input)
        input = torch.sign(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # import pdb; pdb.set_trace()
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)
        self.layer_type = 'FConv2d'
        self.transform = None

    def forward(self, x):
        restore_w = self.weight
        max = restore_w.data.max()
        weight_q = restore_w.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - restore_w).detach() + restore_w

        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'
        self.transform = None

    def forward(self, x):
        restore_w = self.weight
        max = restore_w.data.max()
        weight_q = restore_w.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - restore_w).detach() + restore_w

        return F.linear(x, weight_q, self.bias)


# binary last FC layer
class binary_last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(binary_last_fc, self).__init__(in_features, out_features, bias)

        w = self.weight  # [out_features, in_features]
        sw = w.abs().mean().float().detach()
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)
   
    def forward(self, input):
        a0 = input
        w = self.weight  # [out_features, in_featuers]

        # normalize the weights
        w1 = w - w.mean([1], keepdim=True)  # [out_features, 1]
        w2 = w1 / w1.std([1], keepdim=True)  # [out_features, 1]

        # normalize the input
        a1 = a0 - a0.mean([1], keepdim=True)  # [in_features, 1]
        a2 = a1 / a1.std([1], keepdim=True)  # [in_features, 1]

        # binarize the weights
        bw = XNOR_BinaryQuantize().apply(w2)

        # binarize the input
        ba = XNOR_BinaryQuantize_a().apply(a2)

        # output
        output = F.linear(ba, bw, self.bias)
        output = output * self.alpha
        

        return output


