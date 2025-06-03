from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    GiveUniqueParameterTensors
)

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.streamline import Streamline
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MoveOpPastFork,
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
)

#from finn.transformation.fpgadataflow.make_finegrained import MakeFinegrained

def step_io_surgery(model: ModelWrapper, cfg: DataflowBuildConfig):
    # pre-processing: remove input quantization from model
    first_node = model.graph.node[0]
    if first_node.op_type == "MultiThreshold":
        quantized_input_dtype = model.get_tensor_datatype(first_node.output[0])
        # remove nodes up to first Mul (= MT + Add used for input quant)
        new_input_node = model.get_nodes_by_op_type("Mul")[0]
        new_input_tensor = model.get_tensor_valueinfo(new_input_node.input[0])
        old_input_tensor = model.graph.input[0]
        model.graph.input.remove(old_input_tensor)
        model.graph.input.append(new_input_tensor)
        model.graph.value_info.remove(new_input_tensor) # remove redundant value_info
        new_input_index = model.get_node_index(new_input_node)
        del model.graph.node[0:new_input_index]
        # make sure input datatype is set correctly
        model.set_tensor_datatype(model.graph.input[0].name, quantized_input_dtype)

    # post-processing: remove final softmax node if it remains from training
    final_node = model.graph.node[-1]
    if final_node.op_type in ["LogSoftmax", "Softmax"]:
        softmax_in_tensor = model.get_tensor_valueinfo(final_node.input[0])
        softmax_out_tensor = model.get_tensor_valueinfo(final_node.output[0])
        model.graph.output.remove(softmax_out_tensor)
        model.graph.output.append(softmax_in_tensor)
        model.graph.value_info.remove(softmax_in_tensor) # remove redundant value_info
        model.graph.node.remove(final_node)

    # post-processing: append Top-K node
    final_node = model.graph.node[-1]
    if final_node.op_type != "TopK":
        model = model.transform(InsertTopK(k=1))
    return model

def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model

def step_convert_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hls.InferChannelwiseLinearLayer())
    model = model.transform(to_hls.InferLabelSelectLayer())
    model = model.transform(GiveUniqueNodeNames())
    return model

def step_pre_tidy_up_residual(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueParameterTensors())
    return model

def step_streamline_residual(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(MoveLinearPastFork())
    model = model.transform(MoveLinearPastEltwiseAdd())

    # default streamlining steps
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    model = model.transform(MoveTransposePastFork())
    model = model.transform(MoveTransposePastJoinAdd())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(Streamline())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    return model

def step_convert_to_hls_residual(model: ModelWrapper, cfg: DataflowBuildConfig):
    # mostly default steps + dw conv support + residual add support
    mem_mode = cfg.default_mem_mode.value
    if cfg.standalone_thresholds:
        # doing this first causes all threshold layers to be standalone
        model = model.transform(to_hls.InferThresholdingLayer())
    # needed for elementwise (residual) add
    model = model.transform(to_hls.InferAddStreamsLayer())
    # needed for bipolar MatMul layers
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
    # in case of depthwise convolutions
    model = model.transform(to_hls.InferVectorVectorActivation())
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) as standalone threshold
    model = model.transform(to_hls.InferThresholdingLayer())
    # needed for convolutions
    if cfg.force_rtl_conv_inp_gen:
        model = model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
    else:
        model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    # needed for fork of residual connections
    model = model.transform(to_hls.InferDuplicateStreamsLayer())
    model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    return model

# def step_experimentalconv(model: ModelWrapper, cfg: DataflowBuildConfig):
#     model = model.transform(to_hls.InferExperimentalConvAsFC())
#     return model

# def step_experimentalfg(model: ModelWrapper, cfg: DataflowBuildConfig):
#     model = model.transform(MakeFinegrained())
#     model = model.transform(GiveUniqueNodeNames())
#     model = model.transform(GiveReadableTensorNames())
#     return model
