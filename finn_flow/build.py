# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.core.onnx_exec import execute_onnx
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform

# custom steps
from custom_steps import (
    step_io_surgery,
    step_pre_streamline,
    step_convert_final_layers,
    step_pre_tidy_up_residual,
    step_streamline_residual,
    step_convert_to_hls_residual,
    #step_experimentalconv,
    #step_experimentalfg,
)

model_name = "input_for_finn"
model_type = "vgg" # "vgg" or "resnet" (e.g. for BacalhauNet)
print("model_name", model_name)
# which platforms to build the networks for
zynq_platforms = ["VCU118"]#, "RFSoC2x2"]
alveo_platforms = []
platforms_to_build = zynq_platforms + alveo_platforms
# model_file = "models/%s.onnx" % model_name
model_file = "./export/wise-lion-17/text_export_qonnx.onnx"

print("model_file", model_file)
verification_batchsize = 1024
verification_steps = [
            #build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON, # not usable: expected i/o data includes input quantization and topK
            build_cfg.VerificationStepType.TIDY_UP_PYTHON,
            # build_cfg.VerificationStepType.STREAMLINED_PYTHON,
            ##build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM, # not usable due to RTL SWGs
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ]

if verification_steps:
    # generate golden output from input model for later verification
    verify_model = ModelWrapper(model_file)
    verify_model = cleanup_model(verify_model)
    # input = np.load("input_n_%d.npy"%verification_batchsize)
    # TODO: quantize depending on input quant from model instead of using pre-quantized .npy file
    input = np.load("input_n_%d_quant.npy"%verification_batchsize)
    output = np.zeros((verification_batchsize,2))
    for i in range(verification_batchsize):
        input_n1 = input[i,:]
        input_n1 = np.expand_dims(input_n1, axis=0)
        idict = {"global_in" : input_n1}
        #start_node = model.get_nodes_by_op_type("Conv")[0]
        odict = execute_onnx(verify_model, idict, return_full_exec_context = False, start_node = None, end_node = None)
        output_n1 = odict["global_out"]
        output[i,:] = output_n1
    np.save("output_%s.npy"%model_name, output)
    np.save("output_%s_top1.npy"%model_name, np.argmax(output, axis=1))

# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")

# select target clock frequency
def select_clk_period(platform):
        return 10.0

def select_build_steps(model_type):
    if model_type == "vgg":
        return [
            "step_qonnx_to_finn",
            step_io_surgery,
            "step_tidy_up",
            step_pre_streamline,
            "step_streamline",
            "step_convert_to_hw",
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_target_fps_parallelization",
            # "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
        #     # "step_convert_to_hls",
        #     step_convert_final_layers,
        #     "step_create_dataflow_partition",
        #     "step_target_fps_parallelization",
        #     "step_apply_folding_config",
        #     "step_minimize_bit_width",
        #     "step_generate_estimate_reports",
        #     # "step_hls_codegen",
        #     # "step_hls_ipgen",
        #     "step_set_fifo_depths",
        #     "step_create_stitched_ip",
        #     "step_measure_rtlsim_performance",
        #     "step_out_of_context_synthesis",
        #     "step_synthesize_bitfile",
        #     "step_make_pynq_driver",
        #     "step_deployment_package",
        ]
    elif model_type == "resnet":
        return [
            "step_qonnx_to_finn",
            step_io_surgery,
            step_pre_tidy_up_residual,
            "step_tidy_up",
            step_pre_streamline,
            step_streamline_residual,
            step_convert_to_hls_residual,
            step_convert_final_layers,
            "step_create_dataflow_partition",
            "step_target_fps_parallelization",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_hls_codegen",
            "step_hls_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
    else:
        raise Exception("No build steps defined for this model type")

for platform_name in platforms_to_build:
    shell_flow_type = platform_to_shell(platform_name)
    if shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        vitis_platform = alveo_default_platform[platform_name]
        #vitis_platform = "xilinx_u280_gen3x16_xdma_1_202211_1" # default is "xilinx_u280_xdma_201920_3"
        release_platform_name = vitis_platform
    else:
        vitis_platform = None
        release_platform_name = platform_name

    cfg = build_cfg.DataflowBuildConfig(
        steps = select_build_steps(model_type),
        ### only perform select steps:
        start_step = None,
        stop_step = None,

        ### basics
        output_dir = "output_%.1f_%s_test" % (select_clk_period(platform_name), model_name),
        #output_dir="output_separate_step_%s" % (model_name),
        synth_clk_period_ns = select_clk_period(platform_name),
        board = platform_name,
        shell_flow_type = shell_flow_type,
        vitis_platform = vitis_platform,
        vitis_floorplan_file = None,

        ### debugging
        verbose = True, # prints warnings to stdout
        enable_hw_debug = False,
        enable_build_pdb_debug = False,

        # ### back-end implementation details
        # force_rtl_conv_inp_gen = True,
        # default_mem_mode = build_cfg.ComputeEngineMemMode.CONST,
        # standalone_thresholds = False, # needed (only) for experimental fg flow
        # minimize_bit_width = True,

        ### either define folding config:
        # folding_config_file = "folding_config/Fc32_m_1.json",
        ## or use auto-folding:
        target_fps = 1000000,
        mvau_wwidth_max = 64 * 3 * 4, # C * K * W = MAX_SIMD * W
        folding_two_pass_relaxation = False,

        ### FIFOs
        auto_fifo_depths = True,
        auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM, # CHARACTERIZE or LARGEFIFO_RTLSIM
        large_fifo_mem_style = build_cfg.LargeFIFOMemStyle.AUTO,
    
        ### simulation & verification
        rtlsim_batch_size = 10, # must be > 1 for reasonable measurement of stable-state throughput
        verify_steps=verification_steps,
        verify_input_npy = "input_n_%d_quant_float.npy"%verification_batchsize,
        verify_expected_output_npy = "output_%s_top1.npy"%model_name,
        verify_save_full_context = False, # by default, only the top-level graph output is saved
        verify_save_rtlsim_waveforms = False, # save .vcd waveforms under reports
        rtlsim_use_vivado_comps = True,

        ### enable extra performance optimizations for Vitis flow
        vitis_opt_strategy = build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,

        ### output products
        stitched_ip_gen_dcp = False, # runs synthesis to generate a .dcp for the stitched-IP output product
        generate_outputs = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.OOC_SYNTH,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            #build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)
