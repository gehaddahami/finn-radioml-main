
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.core.onnx_exec import execute_onnx

if 'DATASET_PATH_RADIOML' in os.environ:
    dataset_path = os.environ['DATASET_PATH_RADIOML'] + "/2018/GOLD_XYZ_OSC.0001_1024.hdf5"
    if not os.path.isfile(dataset_path):
        print("ERROR: Dataset not found")
else:
    print("ERROR: DATASET_PATH_RADIOML not set")

# Prepare data loader
class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path):
        super(radioml_18_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]
        self.len = self.data.shape[0]

        # load complete dataset into memory
        self.loaded_data = self.data[:]

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        torch.manual_seed(2018) #seed for random sampler

        train_indices = []
        val_indices = []
        test_indices_overall = []
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
                
                test_indices_overall.extend(test_indices_subclass)

        self.test_sampler_overall = torch.utils.data.SubsetRandomSampler(test_indices_overall)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        #return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

        return self.loaded_data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

def quantize(data):
    quant_min = -2.0
    quant_max = 2.0
    quant_range = quant_max - quant_min
    data_quant = (data - quant_min) / quant_range
    data_quant = np.round(data_quant * 256) - 128
    data_quant = np.clip(data_quant, -128, 127)
    data_quant = data_quant.astype(np.int8)
    return data_quant

model = ModelWrapper("qonnx_exported_model.onnx")
model = cleanup_model(model)
batch_size = 1
num_batches = 1


dataset = radioml_18_dataset(dataset_path)
data_loader_test_overall = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler_overall)
mod_classes = dataset.mod_classes
snr_classes = dataset.snr_classes

batch = 0
y_exp = np.empty((0))
y_snr = np.empty((0))
y_pred = np.empty((0,len(mod_classes)))
for (input, target, snr) in data_loader_test_overall:
    print("processing batch %d.."%batch)

    input = np.array(input)
    np.save("input_n_%d.npy"%batch_size, input)
    np.save("input_n_%d_quant.npy"%batch_size, quantize(input))
    np.save("input_n_%d_quant_float.npy"%batch_size, quantize(input).astype(np.float32))

    #input = quantize(input).astype(np.float32)
    idict = {"global_in" : input}
    
    #start_node = model.get_nodes_by_op_type("Conv")[0]
    odict = execute_onnx(model, idict, return_full_exec_context = False, start_node = None, end_node = None)
    output = odict["global_out"]

    np.save("output_n_%d.npy"%batch_size, output)
    np.save("output_n_%d_top1.npy"%batch_size, int(np.argmax(output)))

    y_pred = np.concatenate((y_pred, output))
    y_exp = np.concatenate((y_exp, target))
    y_snr = np.concatenate((y_snr, snr))

    batch = batch + 1
    if batch == num_batches:
        break

# confusion matrices and accuracy
conf = np.zeros([len(snr_classes),len(mod_classes),len(mod_classes)])
confnorm = np.zeros([len(snr_classes),len(mod_classes),len(mod_classes)])
confnorm_overall = np.zeros([len(mod_classes),len(mod_classes)])
accuracy_per_SNR = []
for snr_idx, snr in enumerate(snr_classes):
    indices_snr = (y_snr == snr).nonzero()
    y_exp_i = y_exp[indices_snr]
    y_pred_i = y_pred[indices_snr]
    for i in range(len(y_exp_i)):
        j = int(y_exp_i[i])
        k = int(np.argmax(y_pred_i[i,:]))
        conf[snr_idx, j,k] = conf[snr_idx, j,k] + 1
    for i in range(0,len(mod_classes)):
        confnorm[snr_idx, i,:] = conf[snr_idx, i,:] / np.sum(conf[snr_idx, i,:])
    accuracy_per_SNR.append(np.sum(np.diag(conf[snr_idx,:,:])) / len(y_exp_i))
conf_overall = np.sum(conf, axis=0)
for i in range(0,len(mod_classes)):
    confnorm_overall[i,:] = conf_overall[i,:] / np.sum(conf_overall[i,:])

accuracy_overall = np.sum(np.diag(conf_overall)) / len(y_exp)
accuracy_30dB = accuracy_per_SNR[-1]

print("Accuracy overall: %f" % accuracy_overall)
print("Accuracy @ 30 dB: %f" % accuracy_30dB)
