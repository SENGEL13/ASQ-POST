import torch.nn as nn
from asq import *
from lsq import *


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, bit: int):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.first_layer = True
        self.quant_module_refactor(self.model, bit)
        

    def quant_module_refactor(self, module: nn.Module, bit: int):
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.Conv2d):
                if self.first_layer:
                    quant_tmp_conv = AsqBnConv2dUnfold(bit=8, first_layer=True)
                    #quant_tmp_conv = LsqBnFirstConv2dUnfold(bit=8)
                    quant_tmp_conv.set_param(child_module)
                    setattr(module, name, quant_tmp_conv)
                    self.first_layer = False
                else:
                    quant_tmp_conv = AsqBnConv2dUnfold(bit=bit)
                    #quant_tmp_conv = LsqBnConv2dUnfold(bit=bit)
                    quant_tmp_conv.set_param(child_module)
                    setattr(module, name, quant_tmp_conv)
                # prev_quantmodule = getattr(module, name)
            if isinstance(child_module, nn.Linear):
                quant_tmp_linear = LsqFixedLinear(bit=8)
                #quant_tmp_linear = AsqLinear(bit=8)
                quant_tmp_linear.set_param(child_module)
                setattr(module, name, quant_tmp_linear)
            else:
                self.quant_module_refactor(child_module, bit)

    def forward(self, input):
        return self.model(input)
