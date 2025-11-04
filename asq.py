import torch.nn as nn
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

class Adapter(nn.Module):
    def __init__(self,in_planes, out_planes, ratio=4):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        
        #assert in_planes>ratio
        if in_planes>=(2*ratio):
            hidden_planes=in_planes//ratio
            self.net=nn.Sequential(
                nn.Linear(in_planes, hidden_planes),
                nn.ReLU(),
                nn.Linear(hidden_planes, 1),
                nn.ReLU(),
            )
            self.layer_num = 2
        else:
            self.net=nn.Sequential(
                nn.Linear(in_planes, 1),
                nn.ReLU(),
            )
            self.layer_num = 1
        self.out_planes = out_planes
        self.initialize_weights()
        
    def initialize_weights(self):
        if self.layer_num==2:
            nn.init.normal_(self.net[2].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.net[2].bias, 1.)
        else:
            nn.init.normal_(self.net[0].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.net[0].bias, 1.)
        #nn.init.constant_(self.net[2].weight, 1e-8)
        #nn.init.constant_(self.out.bias, init_s)
        #device = self.net[2].bias.device
        #self.net[2].bias.data = nn.Parameter(1., requires_grad=True)
        #print(self.out.bias.shape)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=att.view(x.shape[0], -1)
        att=self.net(att) #bs,out_planes+1
        att=torch.clamp(att, min=1e-8)
        return att.view(-1,1,1,1)#, att[:,1:].view(-1,self.out_planes,1,1,1)

class Linear_Adapter(nn.Module):
    def __init__(self,in_planes, ratio=4):#out_planes, 
        super().__init__()
        #assert in_planes>ratio
        if in_planes>=(2*ratio):
            hidden_planes=in_planes//ratio
            self.net=nn.Sequential(
                nn.Linear(in_planes, hidden_planes),
                nn.ReLU(),
                nn.Linear(hidden_planes, 1),
                nn.ReLU(),
            )
            self.layer_num = 2
        else:
            self.net=nn.Sequential(
                nn.Linear(in_planes, 1),
                nn.ReLU(),
            )
            self.layer_num = 1
        
    def initialize_weights(self):
        if self.layer_num==2:
            nn.init.normal_(self.net[2].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.net[2].bias, 1.)
        else:
            nn.init.normal_(self.net[0].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.net[0].bias, 1.)
        #nn.init.constant_(self.net[2].weight, 1e-8)
        #nn.init.constant_(self.out.bias, init_s)
        #device = self.net[2].bias.device
        #self.net[2].bias.data = nn.Parameter(1., requires_grad=True)

    def forward(self,x):
        att=self.net(x) #bs,out_planes+1
        att=torch.clamp(att, min=1e-8)
        return att.view(-1,1)#.view(-1,1,1,1)#, att[:,1:].view(-1,self.out_planes,1,1,1)

# def LSQFakeQuant(grad_scale=None):
#     class _uq(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, x, scale, thd_neg, thd_pos, beta=None):
#             scale_shape = scale.shape
#             if beta is not None:
#                 s_scale = scale * beta
#             else:
#                 s_scale = scale
#             x = x / s_scale
#             #x = round_pass(x)
#             x_c = torch.clamp(x, thd_neg, thd_pos)
#             x_q = x_c.round()
#             ctx.save_for_backward(x, x_q, scale, beta)
#             ctx.other = thd_neg, thd_pos, scale_shape
#             x_q = x_q * s_scale
#             return x_q

#         @staticmethod
#         def backward(ctx, grad_output):
#             #grad_input = grad_output.clone()  # calibration: grad for weights will not be clipped
#             x, x_q, scale, beta = ctx.saved_tensors
#             thd_neg, thd_pos, scale_shape = ctx.other
#             i_neg = (x < thd_neg).float()
#             i_pos = (x > thd_pos).float()
#             # if len(scale_shape)==1:
#             #     grad_s = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum().unsqueeze(dim=0)
#             # else:
#             grad_sb = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum(dim=list(range(1, x.dim())), keepdim=True)

#             if beta is not None:
#                 grad_s = grad_sb * beta
#                 grad_beta = grad_sb * scale
#             else:
#                 grad_s = grad_sb
#                 grad_beta = None
#             if grad_scale:
#                 grad_s = grad_s * grad_scale
#             grad_input = (1 - i_neg - i_pos) * grad_output.clone()
#             return grad_input, grad_s, None, None, grad_beta

#     return _uq().apply

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
                if beta is not None:
                    grad_sb = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum(dim=list(range(1, x.dim())), keepdim=True)
                    grad_s = (grad_sb*beta).sum().unsqueeze(dim=0)
                    grad_beta = grad_sb*scale
                else:
                    grad_s = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum().unsqueeze(dim=0)
                    grad_beta = None
            else:
                grad_s = (grad_output * (i_neg * thd_neg + i_pos * thd_pos + (x_q - x) * (1 - i_neg - i_pos))).sum(dim=list(range(1, x.dim())), keepdim=True)
                grad_beta = None
            if grad_scale:
                grad_s = grad_s * grad_scale
            grad_input = (1 - i_neg - i_pos) * grad_output.clone()
            return grad_input, grad_s, None, None, grad_beta

    return _uq().apply

class AsqWeightQuant(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, uq=False):
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
    
class AsqActQuant(Quantizer):
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
            
    def forward(self, x, b):
        self.s.data.abs_()
        self.s.data.clamp_(min=self.eps)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()/x.shape[0]) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        if self.uq:
            x = LSQFakeQuant(s_grad_scale)(x, self.s, self.thd_neg, self.thd_pos, b)
        else:
            s_scale = grad_scale(self.s, s_grad_scale)
            s_scale = s_scale*b
            x = x / s_scale
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
        return x

# class QuantFunction(nn.Module):
#     def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
#         super().__init__()
#         if all_positive:
#             assert not symmetric, "Positive quantization cannot be symmetric"
#             self.thd_neg = 0
#             self.thd_pos = 2 ** bit - 1
#         else:
#             if symmetric:
#                 self.thd_neg = - 2 ** (bit - 1) + 1
#                 self.thd_pos = 2 ** (bit - 1) - 1
#             else:
#                 self.thd_neg = - 2 ** (bit - 1)
#                 self.thd_pos = 2 ** (bit - 1) - 1
#         self.per_channel = per_channel
#     def get_init_scale(self, x):
#         if self.per_channel:
#             return x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5)
#         else:
#             return x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            
#     def forward(self, x, scale):
#         if self.per_channel:
#             s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
#         else:
#             s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
#         s_scale = grad_scale(scale, s_grad_scale)
#         x = x / s_scale
#         x = torch.clamp(x, self.thd_neg, self.thd_pos)
#         x = round_pass(x)
#         x = x * s_scale
#         return x

class AsqBnConv2dUnfold(Module):
    def __init__(self, bit=4, first_layer=False, per_channel=True): 
        super(AsqBnConv2dUnfold, self).__init__()
        self.bit = bit
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.ones(1))
        self.first_layer = first_layer
        
    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn
        if self.first_layer:
            self.adapter = Adapter(conv.in_channels, conv.out_channels)
            self.quan_w_fn = AsqWeightQuant(bit=self.bit)
            self.quan_a_fn = AsqActQuant(bit=self.bit, all_positive=False)
        else:
            self.adapter = Adapter(conv.in_channels, conv.out_channels)
            self.quan_w_fn = AsqWeightQuant(bit=self.bit)
            self.quan_a_fn = AsqActQuant(bit=self.bit)
        self.quan_w_fn.init_from(self.conv.weight) 

    def __repr__(self):
        conv_s = super(AsqBnConv2dUnfold, self).__repr__()
        s = "({0}, bit={1}, groups={2}, wt-channel-wise={3}".format(
            conv_s, self.bit, self.conv.groups, self.per_channel)
        return s

    def forward(self, x):
        if self.init_state:
            self.adapter.initialize_weights()
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        b_a = self.adapter(x)#, b_w
        quantized_weight = self.quan_w_fn(self.conv.weight)
        quantized_act = self.quan_a_fn(x, b_a)
        return self.bn(F.conv2d(quantized_act, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))

class AsqBnFirstConv2dUnfold(Module):
    def __init__(self, bit=4, per_channel=False):
        super(AsqBnFirstConv2dUnfold, self).__init__()
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
        conv_s = super(AsqBnFirstConv2dUnfold, self).__repr__()
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

class AsqConv2d(Module):
    def __init__(self, bit=4, first_layer=False, per_channel=True): 
        super(AsqConv2d, self).__init__()
        self.bit = bit
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.ones(1))
        self.first_layer = first_layer
        
    def set_param(self, conv):
        self.conv = conv
        if self.first_layer:
            self.adapter = Adapter(conv.in_channels, conv.out_channels)
            self.quan_w_fn = AsqWeightQuant(bit=self.bit)
            self.quan_a_fn = AsqActQuant(bit=self.bit, all_positive=False)
        else:
            self.adapter = Adapter(conv.in_channels, conv.out_channels)
            self.quan_w_fn = AsqWeightQuant(bit=self.bit)
            self.quan_a_fn = AsqActQuant(bit=self.bit)
        self.quan_w_fn.init_from(self.conv.weight) 

    def __repr__(self):
        conv_s = super(AsqConv2d, self).__repr__()
        s = "({0}, bit={1}, groups={2}, wt-channel-wise={3}".format(
            conv_s, self.bit, self.conv.groups, self.per_channel)
        return s

    def forward(self, x):
        if self.init_state:
            self.adapter.initialize_weights()
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        b_a = self.adapter(x)#, b_w
        quantized_weight = self.quan_w_fn(self.conv.weight)
        quantized_act = self.quan_a_fn(x, b_a)
        return F.conv2d(quantized_act, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

class AsqLinear(Module):
    def __init__(self, bit=4, per_channel=False):
        super(AsqLinear, self).__init__()

        self.bit = bit
        self.quan_w_fn = AsqWeightQuant(bit=bit)#per_channel=per_channel
        self.quan_a_fn = AsqActQuant(bit=bit)#per_channel=per_channel
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.ones(1))
    def __repr__(self):
        linear_s = super(AsqLinear, self).__repr__()
        s = "({0},bit={1}, per_channel={2}".format(
            linear_s, self.bit, self.per_channel)
        return s
    def set_param(self, linear):
        self.adapter = Linear_Adapter(linear.in_features)
        self.linear = linear
        self.quan_w_fn.init_from(linear.weight)
    def forward(self, x):
        if self.init_state:
            self.adapter.initialize_weights()
            self.quan_a_fn.init_from(x)
            self.init_state.fill_(0)
        b_a = self.adapter(x)#, b_w
        quantized_weight = self.quan_w_fn(self.linear.weight)
        quantized_act = self.quan_a_fn(x, b_a)
        return F.linear(quantized_act, quantized_weight, self.linear.bias)

class FirstConv2dUnfold(Module):
    def __init__(self, bit=4, per_channel=False):
        super(FirstConv2dUnfold, self).__init__()

    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn

    def forward(self, x):
        # quantized_weight = self.conv.weight#self.quan_w_fn(self.conv.weight)
        # quantized_act = x#self.quan_a_fn(x)
        return self.bn(self.conv(x))

class LastLinear(Module):
    def __init__(self, bit=4, per_channel=False):
        super(LastLinear, self).__init__()

    def set_param(self, linear):
        self.linear = linear
    def forward(self, x):
        # quantized_weight = self.linear.weight#self.quan_w_fn(self.linear.weight)
        # quantized_act = x#self.quan_a_fn(x)
        return self.linear(x)
    
# class AsqBnConv2dUnfold(Module):
#     def __init__(self, bit=4, first_layer=False, per_channel=True): 
#         super(AsqBnConv2dUnfold, self).__init__()
#         self.bit = bit
#         self.per_channel = per_channel
#         self.register_buffer('init_state', torch.ones(1))
#         self.first_layer = first_layer
        
#     def set_param(self, conv, bn):
#         self.conv = conv
#         self.bn = bn
#         if self.first_layer:
#             self.adapter = Adapter(conv.in_channels, conv.out_channels, ratio=1)
#             self.quan_w_fn = AsqWeightQuant(bit=self.bit)
#             self.quan_a_fn = AsqActQuant(bit=self.bit, all_positive=False)
#         else:
#             self.adapter = Adapter(conv.in_channels, conv.out_channels)
#             self.quan_w_fn = AsqWeightQuant(bit=self.bit)
#             self.quan_a_fn = AsqActQuant(bit=self.bit)
#         self.quan_w_fn.init_from(self.conv.weight) 

#     def __repr__(self):
#         conv_s = super(AsqBnConv2dUnfold, self).__repr__()
#         s = "({0}, bit={1}, groups={2}, wt-channel-wise={3}".format(
#             conv_s, self.bit, self.conv.groups, self.per_channel)
#         return s

#     def forward(self, x):
#         if self.init_state:
#             self.adapter.initialize_weights()
#             self.quan_a_fn.init_from(x)
#             self.init_state.fill_(0)
#         b_a, b_w = self.adapter(x)
#         quantized_weight = self.quan_w_fn(self.conv.weight, b_w)
#         quantized_act = self.quan_a_fn(x, b_a)
#         #print(quantized_act.shape, quantized_weight.shape)
#         bs, c, h, w = quantized_act.shape
#         bs, o_p, i_p, kh, kw = quantized_weight.shape
#         output = F.conv2d(quantized_act.view(1, bs*c, h, w), quantized_weight.view(bs * o_p, i_p, kh, kw), self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, bs)
#         return self.bn(output.view(bs, o_p, output.shape[-2], output.shape[-1]))