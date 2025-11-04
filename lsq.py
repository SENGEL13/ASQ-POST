import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def LSQFakeQuant(grad_scale=None):
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale, thd_neg, thd_pos, beta=None):
            scale_shape = scale.shape
            if beta is not None:
                s_scale = scale * beta
            else:
                s_scale = scale
            x = x / s_scale
            #x = round_pass(x)
            x_c = torch.clamp(x, thd_neg, thd_pos)
            x_q = x_c.round()
            ctx.save_for_backward(x, x_q, scale, beta)
            ctx.other = thd_neg, thd_pos, scale_shape
            x_q = x_q * s_scale
            return x_q

        @staticmethod
        def backward(ctx, grad_output):
            #grad_input = grad_output.clone()  # calibration: grad for weights will not be clipped
            x, x_q, scale, beta = ctx.saved_tensors
            thd_neg, thd_pos, scale_shape = ctx.other
            i_neg = (x < thd_neg).float()
            i_pos = (x > thd_pos).float()
            if len(scale_shape)==1:
                grad_s = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum().unsqueeze(dim=0)
            else:
                grad_s = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum(dim=list(range(1, x.dim())), keepdim=True)

            if beta is not None:
                grad_s = grad_s * beta
                grad_beta = grad_s * scale
            else:
                grad_beta = None
            if grad_scale:
                grad_s = grad_s * grad_scale
            grad_input = (1 - i_neg - i_pos) * grad_output.clone()
            return grad_input, grad_s, None, None, grad_beta

    return _uq().apply

class LsqWeightQuant(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, uq=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        self.eps = torch.finfo(torch.float32).eps
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))
        self.uq = uq

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            mean = x.detach().mean()
            std = x.detach().std()
            s_init = torch.max((mean-3*std).abs(), (mean+3*std).abs())/2**(bit_width-1)
            self.s.data.copy_(s_init)
            
    def forward(self, x):
        self.s.data.abs_()
        self.s.data.clamp_(min=self.eps)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()/x.shape[0]) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        if self.uq:
            x = LSQFakeQuant(s_grad_scale)(x, self.s, self.thd_neg, self.thd_pos)
        else:
            s_scale = grad_scale(self.s, s_grad_scale)
            #s_scale = s_scale*b
            #print(x.device, s_scale.device)
            x = x / s_scale
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
        return x

    
class LsqActQuant(Quantizer):
    def __init__(self, bit, all_positive=True, symmetric=False, per_channel=False, uq=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.eps = torch.finfo(torch.float32).eps
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))
        self.uq = uq

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s.data.copy_(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            
    def forward(self, x):
        self.s.data.abs_()
        self.s.data.clamp_(min=self.eps)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        if self.uq:
            x = LSQFakeQuant(s_grad_scale)(x, self.s, self.thd_neg, self.thd_pos)
        else:
            s_scale = grad_scale(self.s, s_grad_scale)
            #s_scale = s_scale*b
            #print(x.device, s_scale.device)
            x = x / s_scale
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
        return x
        
class LsqConv2d(Module):
    def __init__(self, bit=4, per_channel=False):
        super(LsqConv2d, self).__init__()
        self.bit = bit
        self.quan_w_fn = LsqWeightQuant(bit=bit)#per_channel=per_channel
        self.quan_a_fn = LsqActQuant(bit=bit)#per_channel=per_channel
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.ones(1))
    def __repr__(self):
        conv_s = super(LsqConv2d, self).__repr__()
        s = "({0},bit={1}, per_channel={2}".format(
            conv_s, self.bit, self.per_channel)
        return s
    def set_param(self, conv):
        self.conv = conv
        self.quan_w_fn.init_from(self.conv.weight) 
        
    def forward(self, x):
        if self.init_state:
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        quantized_weight = self.quan_w_fn(self.conv.weight)
        quantized_act = self.quan_a_fn(x)
        return F.conv2d(quantized_act, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

class LsqFixedLinear(Module):
    def __init__(self, bit=4, per_channel=False):
        super(LsqFixedLinear, self).__init__()

        # self.bit = bit
        # self.quan_w_fn = LsqWeightQuant(bit=bit)#per_channel=per_channel
        # self.quan_a_fn = LsqActQuant(bit=bit)#per_channel=per_channel
        # self.per_channel = per_channel
        # self.register_buffer('init_state', torch.ones(1))
    def __repr__(self):
        linear_s = super(LsqFixedLinear, self).__repr__()
        # s = "({0},bit={1}, per_channel={2}".format(
        #     linear_s, self.bit, self.per_channel)
        # return s
    def set_param(self, linear):
        self.linear = linear
        # self.quan_w_fn.init_from(linear.weight)
    def forward(self, x, if_init_act=False):
        # if self.init_state:
        #     self.quan_a_fn.init_from(x)
        #     self.init_state.fill_(0)
        quantized_weight = self.linear.weight#self.quan_w_fn(self.linear.weight)
        quantized_act = x#self.quan_a_fn(x)
        return F.linear(quantized_act, quantized_weight, self.linear.bias)

class LsqBnConv2d(Module):
    def __init__(self, bit=4, per_channel=True, first_layer=False):
        super(LsqBnConv2d, self).__init__()
        self.bit = bit
        self.quan_w_fn = LsqWeightQuant(bit=bit)#per_channel=per_channel
        ###
        self.first_layer = first_layer
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.ones(1))

    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn
        if self.first_layer:
            self.quan_a_fn = LsqActQuant(bit=self.bit, all_positive=False)
        else:
            self.quan_a_fn = LsqActQuant(bit=self.bit)#per_channel=per_channel

    def __repr__(self):
        conv_s = super(LsqBnConv2d, self).__repr__()
        s = "({0}, bit={1}, groups={2}, wt-channel-wise={3}".format(
            conv_s, self.bit, self.conv.groups, self.per_channel)
        return s

    def forward(self, x, if_init_act=False):
        running_std = torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.conv.weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])
        if self.conv.bias is not None:
            scaled_bias = self.conv.bias
        else:
            scaled_bias = torch.zeros_like(self.bn.running_mean)
        scaled_bias = (scaled_bias - self.bn.running_mean.detach()) * scale_factor + self.bn.bias
        if self.init_state:
            self.quan_w_fn.init_from(scaled_weight)
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        quantized_weight = self.quan_w_fn(scaled_weight)
        quantized_act = self.quan_a_fn(x)
        return F.conv2d(quantized_act, quantized_weight, scaled_bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
    
class LsqBnConv2dUnfold(Module):
    def __init__(self, bit=4, per_channel=False):
        super(LsqBnConv2dUnfold, self).__init__()
        self.bit = bit
        self.quan_w_fn = LsqWeightQuant(bit=bit)#per_channel=per_channel
        self.quan_a_fn = LsqActQuant(bit=bit)#per_channel=per_channel
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.ones(1))

    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn
        self.quan_w_fn.init_from(self.conv.weight) 

    def __repr__(self):
        conv_s = super(LsqBnConv2dUnfold, self).__repr__()
        s = "({0}, bit={1}, groups={2}, wt-channel-wise={3}".format(
            conv_s, self.bit, self.conv.groups, self.per_channel)
        return s

    def forward(self, x):
        if self.init_state:
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        quantized_weight = self.quan_w_fn(self.conv.weight)
        quantized_act = self.quan_a_fn(x)
        return self.bn(F.conv2d(quantized_act, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))
    
class LsqBnFirstConv2dUnfold(Module):
    def __init__(self, bit=4, per_channel=False):
        super(LsqBnFirstConv2dUnfold, self).__init__()
        # self.bit = bit
        # self.quan_w_fn = LsqWeightQuant(bit=bit)#per_channel=per_channel
        # self.quan_a_fn = LsqActQuant(all_positive=False, bit=bit)#per_channel=per_channel
        # self.per_channel = per_channel
        # self.register_buffer('init_state', torch.ones(1))

    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn
        # self.quan_w_fn.init_from(self.conv.weight) 

    def __repr__(self):
        conv_s = super(LsqBnFirstConv2dUnfold, self).__repr__()
        # s = "({0}, bit={1}, groups={2}, wt-channel-wise={3}".format(
        #     conv_s, self.bit, self.conv.groups, self.per_channel)
        # return s

    def forward(self, x):
        # if self.init_state:
        #     self.quan_a_fn.init_from(x)
        #     self.init_state.fill_(0)
        quantized_weight = self.conv.weight#self.quan_w_fn(self.conv.weight)
        quantized_act = x#self.quan_a_fn(x)
        return self.bn(F.conv2d(quantized_act, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))