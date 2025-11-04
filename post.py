# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import torch.nn as nn
import torch
import torch.nn.functional as F
from lsq import *
from asq import *

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


# this function construct an additive pot quantization levels set, with clipping threshold = 1,
def build_power_value(B=2):
    values = [0.]
    base = 2#**0.5
    for i in range(2 ** B - 1):
        values.append(-base ** (-i - 1))
        values.append(base ** (-i - 1))
    values.append(-1)
    values.append(1)
    values = torch.Tensor(list(set(values)))
    return values

# build_power_value(3)


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def post_quantization(tensor, alpha, proj_set, thd_neg, thd_pos, grad_scale=None):
    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        # sign = x.sign()
        value_s = value_s.type_as(x)
        # xhard = xhard.abs()
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)#.mul(sign)
        xhard = xhard
        xout = (xhard - x).detach() + x
        return xout

    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    data = data.clamp(thd_neg, thd_pos)
    data_q = power_quant(data, proj_set)
    data_q = data_q * alpha
    return data_q


def uq_with_calibrated_graditens(grad_scale=None):
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)
            input_c = input.clamp(min=-1, max=1)
            input_q = input_c.round()
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # calibration: grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            if grad_scale:
                grad_alpha = grad_alpha * grad_scale
            return grad_input, grad_alpha

    return _uq().apply


def uniform_quantization(tensor, alpha, bit, is_weight=True, grad_scale=None):
    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data = data * (2 ** (bit - 1) - 1)
        data_q = (data.round() - data).detach() + data
        data_q = data_q / (2 ** (bit - 1) - 1) * alpha
    else:
        data = data.clamp(0, 1)
        data = data * (2 ** bit - 1)
        data_q = (data.round() - data).detach() + data
        data_q = data_q / (2 ** bit - 1) * alpha
    return data_q

class PostWeightQuant(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        self.thd_neg = -1
        self.thd_pos = 1#2**(-1)
        # print(self.thd_neg, self.thd_pos)
        self.proj_set_weight = build_power_value(B=bit - 1)
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))
        self.bit = bit

    def init_from(self, x, *args, **kwargs):
        shape = x.shape
        if self.per_channel:
            # self.s = nn.Parameter(
            #     x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            self.s = nn.Parameter(
                torch.quantile(x.detach().abs().view(shape[0],-1), 0.99, dim=1, keepdim=True).view(shape[0],1,1,1))
        else:
            # self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.s = nn.Parameter(torch.quantile(x.detach().abs(), 0.99))
            
    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()/x.shape[0]) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        x = post_quantization(x, self.s, self.proj_set_weight, self.thd_neg, self.thd_pos, s_grad_scale)
        return x
class PostBnConv2d(Module):
    def __init__(self, bit=5, adaptive=True, per_channel=True):
        super(PostBnConv2d, self).__init__()
        self.bit = bit
        self.per_channel = per_channel
        self.adaptive = adaptive
        # self.weight_alpha = torch.nn.Parameter(torch.tensor(3.0))
        self.quan_w_fn = PostWeightQuant(bit=bit)
        if adaptive:
            self.quan_a_fn = AsqActQuant(bit=bit)
        else:
            self.quan_a_fn = LsqActQuant(bit=bit)
        self.register_buffer('init_state', torch.ones(1))

    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn
        if self.adaptive:
            self.adapter = Adapter(conv.in_channels, conv.out_channels)
        self.quan_w_fn.init_from(self.conv.weight) 

    def __repr__(self):
        conv_s = super(PostBnConv2d, self).__repr__()
        s = "({0}, bit={1}, wt-channel-wise={2}".format(
            conv_s, self.bit, self.per_channel)
        return s
    
    def forward(self, x):
        if self.init_state:
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        if self.adaptive:
            beta = self.adapter(x)
            quantized_act = self.quan_a_fn(x, beta)
        else:
            quantized_act = self.quan_a_fn(x)
        quantized_weight = self.quan_w_fn(self.conv.weight)
        return self.bn(F.conv2d(quantized_act, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))

    def show_params(self):
        if self.bit != 32:
            wgt_alpha = round(self.weight_alpha.data.item(), 3)
            print('clipping threshold weight alpha: {:2f}'.format(wgt_alpha))