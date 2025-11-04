from lsq import *
from asq import *
from post import *

class Q_ResNet18_PA(nn.Module):
    def __init__(self, model, bit=4):
        super().__init__()
        # features = getattr(model, 'features')
        # init_block = getattr(features, 'init_block')

        self.quant_init_block_convbn = FirstConv2dUnfold()
        self.quant_init_block_convbn.set_param(model.conv1, model.bn1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for layer_num in range(0, 4):
            layer = getattr(model, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResBlockBn_PA(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'fc')
        self.quant_output = LastLinear(bit=8)
        self.quant_output.set_param(output)
        
    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.pool(x)

        x = self.act(x)

        for layer_num in range(0, 4):
            for unit_num in range(0, self.channel[layer_num]):
                tmp_func = getattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x

class Q_ResNet18_Asq(nn.Module):
    def __init__(self, model, bit=4):
        super().__init__()
        # features = getattr(model, 'features')
        # init_block = getattr(features, 'init_block')

        # self.quant_init_block_convbn = AsqBnConv2dUnfold(bit=8, first_layer=True)
        self.quant_init_block_convbn = FirstConv2dUnfold()
        self.quant_init_block_convbn.set_param(model.conv1, model.bn1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for layer_num in range(0, 4):
            layer = getattr(model, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResBlockBn_Asq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'fc')
        # self.quant_output = AsqLinear(bit=8)
        self.quant_output = LastLinear()
        self.quant_output.set_param(output)
        
    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.pool(x)

        x = self.act(x)

        for layer_num in range(0, 4):
            for unit_num in range(0, self.channel[layer_num]):
                tmp_func = getattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x


class Q_ResNet50_Asq(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model, bit=4):
        super().__init__()

        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')
        
        # self.quant_init_convbn = AsqBnConv2dUnfold(bit=8, first_layer=True)
        self.quant_init_block_convbn = FirstConv2dUnfold()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn_Asq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'output')
        # self.quant_output = AsqLinear(bit=8)
        self.quant_output = LastLinear()
        #self.quant_output = LsqFixedLinear(bit=8)
        self.quant_output.set_param(output)

    def forward(self, x):
        x = self.quant_init_convbn(x)
        
        x = self.act(x)
        
        x = self.pool(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x

class Q_ResNet50_Lsq(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model, bit=4):
        super().__init__()

        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')
        
        self.quant_init_convbn =  LsqBnFirstConv2dUnfold(bit=8)
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn_Lsq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'output')
        self.quant_output = LsqFixedLinear(bit=8)
        self.quant_output.set_param(output)

    def forward(self, x):
        x = self.quant_init_convbn(x)

        x = self.act(x)

        x = self.pool(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x

class Q_ResNet18_Lsq(nn.Module):
    def __init__(self, model, bit=4):
        super().__init__()
        # features = getattr(model, 'features')
        # init_block = getattr(features, 'init_block')

        self.quant_init_block_convbn = LsqBnFirstConv2dUnfold(bit=8)
        self.quant_init_block_convbn.set_param(model.conv1, model.bn1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for layer_num in range(0, 4):
            layer = getattr(model, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResBlockBn_Lsq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'fc')
        self.quant_output = LsqFixedLinear(bit=8)
        self.quant_output.set_param(output)
        
    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.pool(x)

        x = self.act(x)

        for layer_num in range(0, 4):
            for unit_num in range(0, self.channel[layer_num]):
                tmp_func = getattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x


class Q_ResNet34_Asq(nn.Module):
    def __init__(self, model, bit=4):
        super().__init__()
        # features = getattr(model, 'features')
        # init_block = getattr(features, 'init_block')

        # self.quant_init_block_convbn = AsqBnConv2dUnfold(bit=8, first_layer=True)
        self.quant_init_block_convbn = FirstConv2dUnfold()
        self.quant_init_block_convbn.set_param(model.conv1, model.bn1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for layer_num in range(0, 4):
            layer = getattr(model, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResBlockBn_Asq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'fc')
        # self.quant_output = AsqLinear(bit=8)
        self.quant_output = LastLinear()
        self.quant_output.set_param(output)
        
    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.pool(x)

        x = self.act(x)

        for layer_num in range(0, 4):
            for unit_num in range(0, self.channel[layer_num]):
                tmp_func = getattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x

class Q_ResNet34_Lsq(nn.Module):
    def __init__(self, model, bit=4):
        super().__init__()
        # features = getattr(model, 'features')
        # init_block = getattr(features, 'init_block')

        self.quant_init_block_convbn = LsqBnFirstConv2dUnfold(bit=8)
        self.quant_init_block_convbn.set_param(model.conv1, model.bn1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for layer_num in range(0, 4):
            layer = getattr(model, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResBlockBn_Lsq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        output = getattr(model, 'fc')
        self.quant_output = LsqFixedLinear(bit=8)
        self.quant_output.set_param(output)
        
    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.pool(x)

        x = self.act(x)

        for layer_num in range(0, 4):
            for unit_num in range(0, self.channel[layer_num]):
                tmp_func = getattr(self, f"layer{layer_num + 1}.unit{unit_num + 1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x
    
class Q_ResNet20_Asq(nn.Module):

    def __init__(self, model, bit=4):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_init_block_convbn = AsqBnConv2dUnfold(bit=8, first_layer=True)
        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)

        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn_Asq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=8, stride=1)

        output = getattr(model, 'output')
        self.quant_output = LsqLinear(bit=8)
        self.quant_output.set_param(output)

    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.act(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x, x_tmp


class Q_ResNet20_lsq(nn.Module):

    def __init__(self, model, bit=4):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_init_block_convbn = LsqBnFirstConv2dUnfold(bit=8)
        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)

        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn_Lsq(bit=bit)
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = nn.AvgPool2d(kernel_size=8, stride=1)

        output = getattr(model, 'output')
        self.quant_output = LsqLinear(bit=8)
        self.quant_output.set_param(output)

    def forward(self, x):
        x = self.quant_init_block_convbn(x)

        x = self.act(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x = tmp_func(x)

        x = self.final_pool(x)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x)

        return x

class Q_ResBlockBn_PA(nn.Module):#LSQ
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self, bit = 4):
        super(Q_ResBlockBn_PA, self).__init__()
        self.bit = bit

    def set_param(self, unit):
        self.resize_identity = unit.downsample is not None
        
        self.quant_convbn1 = PostBnConv2d(bit=self.bit)
        self.quant_convbn1.set_param(unit.conv1, unit.bn1)

        self.quant_convbn2 = PostBnConv2d(bit=self.bit)
        self.quant_convbn2.set_param(unit.conv2, unit.bn2)

        if self.resize_identity:
            self.quant_identity_convbn = PostBnConv2d(bit=self.bit)
            self.quant_identity_convbn.set_param(unit.downsample[0], unit.downsample[1])

    def forward(self, x):
        # forward using the quantized modules
        if self.resize_identity:
            identity = self.quant_identity_convbn(x)
        else:
            identity = x

        x = self.quant_convbn1(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn2(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x

class Q_ResBlockBn_Lsq(nn.Module):#LSQ
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self, bit = 4):
        super(Q_ResBlockBn_Lsq, self).__init__()
        self.bit = bit

    def set_param(self, unit):
        self.resize_identity = unit.downsample is not None
        
        self.quant_convbn1 = LsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn1.set_param(unit.conv1, unit.bn1)

        self.quant_convbn2 = LsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn2.set_param(unit.conv2, unit.bn2)

        if self.resize_identity:
            self.quant_identity_convbn = LsqBnConv2dUnfold(bit=self.bit)
            self.quant_identity_convbn.set_param(unit.downsample[0], unit.downsample[1])

    def forward(self, x):
        # forward using the quantized modules
        if self.resize_identity:
            identity = self.quant_identity_convbn(x)
        else:
            identity = x

        x = self.quant_convbn1(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn2(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x

class Q_ResBlockBn_Asq(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self, bit = 4):
        super(Q_ResBlockBn_Asq, self).__init__()
        self.bit = bit

    def set_param(self, unit):
        self.resize_identity = unit.downsample is not None
        
        self.quant_convbn1 = AsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn1.set_param(unit.conv1, unit.bn1)

        self.quant_convbn2 = AsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn2.set_param(unit.conv2, unit.bn2)

        if self.resize_identity:
            self.quant_identity_convbn = AsqBnConv2dUnfold(bit=self.bit)
            self.quant_identity_convbn.set_param(unit.downsample[0], unit.downsample[1])

    def forward(self, x):
        # forward using the quantized modules
        if self.resize_identity:
            identity = self.quant_identity_convbn(x)
        else:
            identity = x

        x = self.quant_convbn1(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn2(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x

class Q_ResUnitBn_Asq(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self, bit=4):
        super(Q_ResUnitBn_Asq, self).__init__()
        self.bit=bit

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        convbn1 = unit.body.conv1
        self.quant_convbn1 = AsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = AsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        convbn3 = unit.body.conv3
        self.quant_convbn3 = AsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = AsqBnConv2dUnfold(bit=self.bit)
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modulesEEEE
        if self.resize_identity:
            identity = self.quant_identity_convbn(x)
        else:
            identity = x

        x = self.quant_convbn1(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn2(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn3(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x

class Q_ResUnitBn_Lsq(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self, bit=4):
        super(Q_ResUnitBn_Lsq, self).__init__()
        self.bit=bit

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        convbn1 = unit.body.conv1
        self.quant_convbn1 = LsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = LsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        convbn3 = unit.body.conv3
        self.quant_convbn3 = LsqBnConv2dUnfold(bit=self.bit)
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = LsqBnConv2dUnfold(bit=self.bit)
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modulesEEEE
        if self.resize_identity:
            identity = self.quant_identity_convbn(x)
        else:
            identity = x

        x = self.quant_convbn1(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn2(x)
        x = nn.ReLU()(x)

        x = self.quant_convbn3(x)

        x = x + identity

        x = nn.ReLU()(x)

        return x

def q_resnet18(model, bit=4):
    net = Q_ResNet18_PA(model, bit=bit)
    return net
def q_resnet34(model, bit=4):
    net = Q_ResNet34_Asq(model, bit=bit)
    return net
def q_resnet50(model, bit=4):
    net = Q_ResNet50_Asq(model, bit=bit)
    return net