import os
import json
import wandb
import torch
import brevitas
from brevitas.export import export_qonnx
import onnx
from qonnx.util.inference_cost import inference_cost

import models_test

print("Tool versions:")
print("  wandb: %s"%wandb.__version__)
print("  torch: %s"%torch.__version__)
print("  brevitas: %s"%brevitas.__version__)
print("  onnx: %s"%onnx.__version__)
#print("  qonnx: %s"%qonnx.__version__) # installed from local repo via "pip install -e REPO_PATH"

api = wandb.Api()

########### Set what to export
export_type = "FINN" # FINN or TensorRT
export_name = "text_export_qonnx"
artifact_name = "VGG7_quant_v1"
artifact_version = "v0"
artifact = api.artifact("dahami-gehad-paderborn-university/RadioML_Finn_updated/" + artifact_name + ":" + artifact_version)
run = artifact.logged_by()
#run = api.run("felixj/RadioML/nko4jex4")
###########

print("Run information:")
print("  name: %s"%run.name)
print("  config: %s"%run.config)
print("  summary: %s"%run.summary)

# load base topology from models.py
config = config_vgg7_quant = {
  "base_topology": "VGG7_quant_v1",
  "batch_size" : 1024,
  "1st_layer_inputs": 128,
  "num_classes": 2,
  "epochs" : 100,
  "lr": 0.001,
  "lr_scheduler": "EXP", # EXP or CAWR
  "lr_scheduler_halflife": 10, # only for EXP: half life in epochs
  "lr_scheduler_t0": 5, # only for CAWR: t_0
  "lr_scheduler_tmult": 1, # only for CAWR: t_mult
  "train_snr_cutoff": 30, # do not train on data below this SNR (in dB)
  "filters_conv": 4,
  "filters_dense": 64,
  "dropout_conv" : 0.5,
  "dropout_dense": 0.5,
  "in_quant_range": 2.0,
  "in_quant_bits": 2,
  "a_bits_l1": 2, #first layer
  "w_bits_l1": 8, #first layer
  "a_bits": 2,
  "w_bits": 8,
  "a_quant_type": "ReLU", # ReLU or Linear
  "a_quant_min": -2.0, # only for Linear
  "a_quant_max": 2.0, # only for Linear
  "w_quant_channelwise_scaling": False,
}

model = getattr(models_test, run.config["base_topology"])(run.config)

# load trained parameters from artifact
dl_dir = "export/" + run.name
export_qonnx_path = dl_dir + "/" + export_name + ".onnx"
artifact.download(dl_dir)
saved_state = torch.load(dl_dir + "/model_state.pth", map_location=torch.device("cpu"))
model.load_state_dict(saved_state)
model.eval()
input_shape = (1, 2, 128)

# export to QONNX format
export_qonnx(model.cpu(), input_t=torch.randn(input_shape), export_path=export_qonnx_path)

# run inference cost calculation
inference_cost(model_filename_or_wrapper    = export_qonnx_path, 
               output_json       = dl_dir + "/inference_cost.json", 
               output_onnx       = dl_dir + "/inference_cost_preprocessed.onnx",
               preprocess        = True, 
               discount_sparsity = False)

inference_cost(model_filename_or_wrapper    = export_qonnx_path, 
               output_json       = dl_dir + "/inference_cost_sparsity.json", 
               output_onnx       = None,
               preprocess        = True, 
               discount_sparsity = True)

# perform export for backend toolchain
if export_type == "FINN":
    # no extra steps needed
    # FINN-flow takes in exported QONNX and performs conversion to FINN-ONNX + network surgery as first steps
    pass
elif export_type == "TensorRT":
    # Export to regular ONNX for TensorRT in various batch sizes
    batch_sizes = [1, 4, 8, 32, 64, 128, 1024, 4096]
    for batch_size in batch_sizes:
        export_model_filename = dl_dir + "/" + export_name + "_" + str(batch_size) + ".onnx"
        dummy_input=torch.randn(batch_size, 2, 1024)
        torch.onnx.export(model.cpu(), dummy_input, export_model_filename, verbose=False)
        # opset_version=11
else:
    print("ERROR: Unknown export type")

print("Done")
