import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
import os
import shutil

# custom steps
from custom_steps import step_pre_streamline, step_convert_final_layers, step_io_surgery


model_name = "radioml_2mods_small"

# which platforms to build the networks for
zynq_platforms = ["ZCU104"]
alveo_platforms = []
platforms_to_build = zynq_platforms + alveo_platforms


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
    return 4.0


# assemble build flow from custom and pre-existing steps
def select_build_steps(platform):
    return [
        "step_qonnx_to_finn",
        step_io_surgery,
        "step_tidy_up",
        step_pre_streamline,
        "step_streamline",
        "step_convert_to_hw",
        step_convert_final_layers,
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        # "step_hw_codegen",
        # # "step_hw_ipgen",
        # "step_set_fifo_depths",
        # "step_create_stitched_ip",
        # "step_measure_rtlsim_performance",
        # "step_out_of_context_synthesis",
        # "step_synthesize_bitfile",
        # "step_make_pynq_driver",
        # "step_deployment_package",
    ]


# create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)

for platform_name in platforms_to_build:
    shell_flow_type = platform_to_shell(platform_name)
    if shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        vitis_platform = alveo_default_platform[platform_name]
        # for Alveo, use the Vitis platform name as the release name
        # e.g. xilinx_u250_xdma_201830_2
        release_platform_name = vitis_platform
    else:
        vitis_platform = None
        # for Zynq, use the board name as the release name
        # e.g. ZCU104
        release_platform_name = platform_name
    platform_dir = "release/%s" % release_platform_name
    os.makedirs(platform_dir, exist_ok=True)

    cfg = build_cfg.DataflowBuildConfig(
        steps=select_build_steps(platform_name),
        output_dir="output_%s_%s" % (model_name, release_platform_name),
        synth_clk_period_ns=select_clk_period(platform_name),
        board=platform_name,
        shell_flow_type=shell_flow_type,
        vitis_platform=vitis_platform,
        # specialize_layers_config_file="specialize_layers_config/%s_specialize_layers.json"
        # % platform_name,
        # folding_config_file="folding_config/%s_folding_config.json" % platform_name,
        split_large_fifos=True,
        standalone_thresholds=True,
        # enable extra performance optimizations (physopt)
        vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            # build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
        ],
    )
    model_file = "./export/wise-lion-17/text_export_qonnx.onnx"
    build.build_dataflow_cfg(model_file, cfg)

    # copy bitfiles and runtime weights into release dir if found
    bitfile_gen_dir = cfg.output_dir + "/bitfile"
    files_to_check_and_copy = [
        "finn-accel.bit",
        "finn-accel.hwh",
        "finn-accel.xclbin",
    ]
    for f in files_to_check_and_copy:
        src_file = bitfile_gen_dir + "/" + f
        dst_file = platform_dir + "/" + f.replace("finn-accel", model_name)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)

    weight_gen_dir = cfg.output_dir + "/driver/runtime_weights"
    weight_dst_dir = platform_dir + "/%s_runtime_weights" % model_name
    if os.path.isdir(weight_gen_dir):
        weight_files = os.listdir(weight_gen_dir)
        if weight_files:
            shutil.copytree(weight_gen_dir, weight_dst_dir)