import tensorrt as trt
import os.path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py

onnx_file_path = "float_s_best_v1_64.onnx"
input_tensor_name = "0"
output_tensor_name = "95"
batch_size = 64
batch_size_calibration = 64

gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print("Using GPU %d" % gpu)
else:
    gpu = None
    print("Using CPU only")

# Check if dataset is present
dataset_path = "/mnt/upb/groups/agce/scratch/felix/datasets/RadioML/2018/GOLD_XYZ_OSC.0001_1024.hdf5"
if os.path.isfile(dataset_path):
    print("Found dataset file")
else:
    print("ERROR: dataset not found")

# Prepare data loader
class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path):
        super(radioml_18_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]
        self.len = self.data.shape[0]

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24): # all modulations (0 to 23)
            for snr_idx in range(0, 26): # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26*4096*mod + 4096*snr_idx
                indices_subclass = list(range(start_idx, start_idx+4096))
                
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096)) 
                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[split:]
                test_indices_subclass = indices_subclass[:split]
                
                # train on >= -6 dB
                #if snr_idx >= 7:

                # calibrate on >= 6 dB (13)
                if snr_idx >=13 and mod not in [17,18]:
                    train_indices.extend(train_indices_subclass[:100])
                # test at 30 dB
                if snr_idx >=25:
                    test_indices.extend(test_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

dataset = radioml_18_dataset(dataset_path)


logger = trt.Logger(trt.Logger.INFO) #WARNING

builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(onnx_file_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    print("ERROR: Could not parse ONNX")


#calibration
class MyCalibrator(trt.IInt8Calibrator):
    #data_loader_calibration = None
    #calibration_iterator = None
    def __init__(self):
        trt.IInt8Calibrator.__init__(self)
        self.data_loader_calibration = DataLoader(dataset, batch_size=batch_size_calibration, sampler=dataset.train_sampler)
        self.calibration_iterator = iter(self.data_loader_calibration)

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
        #LEGACY_CALIBRATION
        #ENTROPY_CALIBRATION
        #ENTROPY_CALIBRATION_2
        #MINMAX_CALIBRATION

    def get_batch_size(self):
        return batch_size_calibration

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            data, labels_mod, labels_snr = next(self.calibration_iterator)
            calibration_tensor = data.cuda()
            return [calibration_tensor.data_ptr()]
        except StopIteration:
        # This signals to TensorRT that there is no calibration data remaining.
            return []
    
    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None



#NUM_IMAGES_PER_BATCH = 5
#batchstream = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
#Int8_calibrator = EntropyCalibrator(["input_tensor_name"], batchstream)

config = builder.create_builder_config()
config.max_workspace_size = 1 << 24 # 16 MiB
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = MyCalibrator()



serialized_engine = builder.build_serialized_network(network, config)

with open("sample.engine", "wb") as f:
    f.write(serialized_engine)


runtime = trt.Runtime(logger)

#with open("sample.engine", "rb") as f:
#    serialized_engine = f.read()

engine = runtime.deserialize_cuda_engine(serialized_engine)

context = engine.create_execution_context()

input_idx = engine[input_tensor_name]
output_idx = engine[output_tensor_name]




data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

buffers = [None] * 2 # 1 input and 1 output

batch_counter = 0
ok = 0
nok = 0

for input_tensor, labels_mod, labels_snr in data_loader_test:
    print(f"Inferring batch {batch_counter}")
    batch_counter = batch_counter + 1
    
    #prepare output buffer
    output_tensor = torch.empty((input_tensor.size(dim=0),24))

    #print(f"Input tensor shape: {input_tensor.size()}")
    #print(f"Input tensor type: {input_tensor.type()}")
    #print(f"Output tensor shape: {output_tensor.size()}")
    #print(f"Output tensor shape: {output_tensor.type()}")

    #move to GPU memory
    input_tensor = input_tensor.cuda()
    output_tensor = output_tensor.cuda()

    #run inference
    buffers[input_idx] = input_tensor.data_ptr()
    buffers[output_idx] = output_tensor.data_ptr()
    context.execute_v2(buffers)
    #context.execute_async_v2(buffers, stream_ptr) # needs CUDA stream

    #read output buffer
    #print("Output tensor:")
    #print(output_tensor)
    #print("prediction:")
    #print(torch.argmax(output_tensor, 1))
    #print("label:")
    #print(labels_mod)

    correct = (torch.argmax(output_tensor.cpu(), 1) == labels_mod).sum()
    ok += int(correct)
    nok += int(torch.numel(labels_mod) - correct)

print(f"ok: {ok}")
print(f"nok: {nok}")
print(f"Accuracy: {ok/(ok+nok)}")