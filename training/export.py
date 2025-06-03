import os
import json
import wandb
import torch
import brevitas
from brevitas.export import export_qonnx
import onnx
from qonnx.util.inference_cost import inference_cost

import models

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
artifact_name = "VGG10_quant_v1"
artifact_version = "v285"
artifact = api.artifact("felixj/RadioML/" + artifact_name + ":" + artifact_version)
run = artifact.logged_by()
#run = api.run("felixj/RadioML/nko4jex4")
############

print("Run information:")
print("  name: %s"%run.name)
print("  config: %s"%run.config)
print("  summary: %s"%run.summary)

# load base topology from models.py
model = getattr(models, run.config["base_topology"])(run.config)

# load trained parameters from artifact
dl_dir = "export/" + run.name
export_qonnx_path = dl_dir + "/" + export_name + ".onnx"
artifact.download(dl_dir)
saved_state = torch.load(dl_dir + "/model_state.pth", map_location=torch.device("cpu"))
model.load_state_dict(saved_state)
model.eval()
input_shape = (1, 2, 1024)

# export to QONNX format
export_qonnx(model.cpu(), input_t=torch.randn(input_shape), export_path=export_qonnx_path)

# run inference cost calculation
inference_cost(model_filename    = export_qonnx_path, 
               output_json       = dl_dir + "/inference_cost.json", 
               output_onnx       = dl_dir + "/inference_cost_preprocessed.onnx",
               preprocess        = True, 
               discount_sparsity = False)

inference_cost(model_filename    = export_qonnx_path, 
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
