from torch import nn
import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.inject.enum import ScalingImplType
import models_bacalhaunetv1
import models_bacalhaunetv1_float
from brevitas.core.quant import QuantType
from functools import partial


# Large model instances such as VGG10, VGG24, Bacalhaunet_v1, Bacalhaunet_v1_float are not used in the paper. Instead, a smaller model with 7 layers is used. 

def VGG7_float_v1 (config):
    return nn.Sequential(
    nn.Conv1d(2, config["filters_conv"], 3, padding=1, bias=False), #1x3 kernel, pad by 1 on each side
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),

    nn.Flatten(),

    nn.Linear(config["1st_layer_inputs"], config["filters_dense"], bias=False),
    nn.BatchNorm1d(config["filters_dense"]),
    nn.Dropout(config["dropout_dense"]),
    nn.ReLU(),

    nn.Linear(config["filters_dense"], config["filters_dense"], bias=False),
    nn.BatchNorm1d(config["filters_dense"]),
    nn.Dropout(config["dropout_dense"]),
    nn.ReLU(),

    nn.Linear(config["filters_dense"], config["num_classes"], bias=True)
    )


def VGG7_quant_v1 (config):
    class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
        min_val = -2.0  #config["in_quant_range"]
        max_val = 2.0  #config["in_quant_range"]
        scaling_impl_type = ScalingImplType.CONST
        quant_type=QuantType.INT
        bit_width = config["in_quant_bits"]
    
    layers = []
    # input quantization
    layers.append(qnn.QuantHardTanh(act_quant=InputQuantizer, return_quant_tensor=True)) 

    # conv block 1
    layers.append(qnn.QuantConv1d(2, config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits_l1"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits_l1"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits_l1"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))

    # conv block 2
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits_l1"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))


    layers.append(nn.Flatten())

    # dense block 1
    layers.append(qnn.QuantLinear(config["1st_layer_inputs"], config["filters_dense"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_dense"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits_l1"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout(config["dropout_dense"]))

    # dense block 2
    layers.append(qnn.QuantLinear(config["filters_dense"], config["filters_dense"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_dense"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits_l1"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout(config["dropout_dense"]))

    # classification layer
    layers.append(qnn.QuantLinear(config["filters_dense"], config["num_classes"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=False, bias=True, bias_quant=IntBias))
    # layers.append(qnn.QuantIdentity(bit_width=config['a_bits'], return_quant_tensor=True))
    

    # unpack list of layers into nn.Sequential
    return nn.Sequential(*layers)
