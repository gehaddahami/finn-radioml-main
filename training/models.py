from torch import nn
import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.inject.enum import ScalingImplType
import models_bacalhaunetv1
import models_bacalhaunetv1_float

def VGG10_float_v1 (config):
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

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
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

    nn.Linear(config["filters_conv"]*8, config["filters_dense"], bias=False),
    nn.BatchNorm1d(config["filters_dense"]),
    nn.Dropout(config["dropout_dense"]),
    nn.ReLU(),

    nn.Linear(config["filters_dense"], config["filters_dense"], bias=False),
    nn.BatchNorm1d(config["filters_dense"]),
    nn.Dropout(config["dropout_dense"]),
    nn.ReLU(),

    nn.Linear(config["filters_dense"], 24, bias=True)
    )

def VGG24_float_v1 (config):
    return nn.Sequential(
    nn.Conv1d(2, config["filters_conv"], 3, padding=1, bias=False), #1x3 kernel, pad by 1 on each side
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),



    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),



    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),



    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),



    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),



    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),



    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.Conv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, bias=False),
    nn.BatchNorm1d(config["filters_conv"]),
    nn.Dropout2d(config["dropout_conv"]),
    nn.ReLU(),

    nn.MaxPool1d(2),
    nn.Flatten(),

    nn.Linear(config["filters_conv"]*8, config["filters_dense"], bias=False),
    nn.BatchNorm1d(config["filters_dense"]),
    nn.Dropout(config["dropout_dense"]),
    nn.ReLU(),

    nn.Linear(config["filters_dense"], config["filters_dense"], bias=False),
    nn.BatchNorm1d(config["filters_dense"]),
    nn.Dropout(config["dropout_dense"]),
    nn.ReLU(),

    nn.Linear(config["filters_dense"], 24, bias=True)
    )

def VGG10_quant_v1 (config):
    class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
        min_val = -config["in_quant_range"]
        max_val = config["in_quant_range"]
        scaling_impl_type = ScalingImplType.CONST
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
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))

    # conv block 3
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))

    # conv block 4
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))

    # conv block 5
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))

    # conv block 6
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))

    # conv block 7
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    layers.append(nn.MaxPool1d(2))
    
    layers.append(nn.Flatten())

    # dense block 1
    layers.append(qnn.QuantLinear(config["filters_conv"]*8, config["filters_dense"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_dense"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout(config["dropout_dense"]))

    # dense block 2
    layers.append(qnn.QuantLinear(config["filters_dense"], config["filters_dense"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_dense"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout(config["dropout_dense"]))

    # classification layer
    layers.append(qnn.QuantLinear(config["filters_dense"], 24, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=False, bias=True, bias_quant=IntBias))
    

    # unpack list of layers into nn.Sequential
    return nn.Sequential(*layers)


def VGG24_quant_v1 (config):
    class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
        min_val = -config["in_quant_range"]
        max_val = config["in_quant_range"]
        scaling_impl_type = ScalingImplType.CONST
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

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    
    layers.append(nn.MaxPool1d(2))

    # conv block 2
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(nn.MaxPool1d(2))

    # conv block 3
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(nn.MaxPool1d(2))

    # conv block 4
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(nn.MaxPool1d(2))

    # conv block 5
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(nn.MaxPool1d(2))

    # conv block 6
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(nn.MaxPool1d(2))

    # conv block 7
    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))

    layers.append(qnn.QuantConv1d(config["filters_conv"], config["filters_conv"], 3, padding=1, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_conv"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout2d(config["dropout_conv"]))
    
    layers.append(nn.MaxPool1d(2))
    
    layers.append(nn.Flatten())

    # dense block 1
    layers.append(qnn.QuantLinear(config["filters_conv"]*8, config["filters_dense"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_dense"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout(config["dropout_dense"]))

    # dense block 2
    layers.append(qnn.QuantLinear(config["filters_dense"], config["filters_dense"], weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=config["w_quant_channelwise_scaling"], bias=False))
    layers.append(nn.BatchNorm1d(config["filters_dense"]))
    if config["a_quant_type"] == "ReLU":
        layers.append(qnn.QuantReLU(bit_width=config["a_bits"], return_quant_tensor=True))
    elif config["a_quant_type"] == "Linear":
        layers.append(qnn.QuantHardTanh(bit_width=config["a_bits"], max_val=config["a_quant_max"], min_val=config["a_quant_min"], return_quant_tensor=True))
    layers.append(nn.Dropout(config["dropout_dense"]))

    # classification layer
    layers.append(qnn.QuantLinear(config["filters_dense"], 24, weight_bit_width=config["w_bits"], weight_scaling_per_output_channel=False, bias=True, bias_quant=IntBias))
    

    # unpack list of layers into nn.Sequential
    return nn.Sequential(*layers)


def Bacalhaunet_v1 (config):

    # Define config (taken from submitted run_config.ods)
    kernels       = [27, 21, 15,  9,  9]
    strides       = [ 2,  1,  2,  1,  2]
    #out_channels  = [24, 24, 90, 90, 256]
    out_channels  = [config["filters_conv1"], config["filters_conv1"], config["filters_conv2"], config["filters_conv2"], config["filters_conv2"]]
    #layers_w_bits = [ 6,  6,  6,  6,  6]
    layers_w_bits = [config["w_bits"], config["w_bits"], config["w_bits"], config["w_bits"], config["w_bits"]]
    #layers_a_bits = [ 6,  6,  6,  6,  6]
    layers_a_bits = [config["a_bits"], config["a_bits"], config["a_bits"], config["a_bits"], config["a_bits"]]

    # Create the layers configurations
    layers = []
    for sub_idx in range(len(kernels)):
        # recast to python int is needed on wbits and abits since brevitas required them to be python int data types
        layers.append(models_bacalhaunetv1.BacalhauNetLayerConfig(kernel=int(kernels[sub_idx]),
                                             stride=int(strides[sub_idx]),
                                             out_channels=int(out_channels[sub_idx]),
                                             w_bits=int(layers_w_bits[sub_idx]),
                                             a_bits=int(layers_a_bits[sub_idx])))

    # Defines the model with the defined layers configurations
    return models_bacalhaunetv1.BacalhauNetV1(
        models_bacalhaunetv1.BacalhauNetConfig(
            in_samples=1024,
            in_channels=2,
            num_classes=24,
            hardtanh_bit_width=8,
            layers=layers,
            pool_bit_width=6,
            dropout_prob=0,
            fc_bit_width=6
        )
    )

def Bacalhaunet_v1_float (config):

    # Define config (taken from submitted run_config.ods)
    kernels       = [27, 21, 15,  9,  9]
    strides       = [ 2,  1,  2,  1,  2]
    #out_channels  = [24, 24, 48, 48, 48]
    out_channels  = [config["filters_conv1"], config["filters_conv1"], config["filters_conv2"], config["filters_conv2"], config["filters_conv2"]]
    layers_w_bits = [ 6,  6,  6,  6,  6] # IGNORED FOR FLOAT
    layers_a_bits = [ 6,  6,  6,  6,  6] # IGNORED FOR FLOAT

    # Create the layers configurations
    layers = []
    for sub_idx in range(len(kernels)):
        # recast to python int is needed on wbits and abits since brevitas required them to be python int data types
        layers.append(models_bacalhaunetv1_float.BacalhauNetLayerConfig(kernel=int(kernels[sub_idx]),
                                             stride=int(strides[sub_idx]),
                                             out_channels=int(out_channels[sub_idx]),
                                             w_bits=int(layers_w_bits[sub_idx]),
                                             a_bits=int(layers_a_bits[sub_idx])))

    # Defines the model with the defined layers configurations
    return models_bacalhaunetv1_float.BacalhauNetV1(
        models_bacalhaunetv1_float.BacalhauNetConfig(
            in_samples=1024,
            in_channels=2,
            num_classes=24,
            hardtanh_bit_width=8,
            layers=layers,
            pool_bit_width=6,
            dropout_prob=0,
            fc_bit_width=6
        )
    )
