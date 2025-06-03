import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import h5py
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.core.onnx_exec import execute_onnx
import matplotlib.pyplot as plt
import seaborn as sns


if 'DATASET_PATH_RADIOML' in os.environ:
    dataset_path = os.environ['DATASET_PATH_RADIOML'] + "/2018/GOLD_XYZ_OSC.0001_1024.hdf5"
    if not os.path.isfile(dataset_path):
        print("ERROR: Dataset not found")
else:
    dataset_path = '/home/gehad/Downloads/dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'

# Prepare data loader
class Radioml_18(Dataset):
    def __init__(self, dataset_path, snr_ratio: int = 0, sequence_length: int = None, selected_modulations=None): 
        super(Radioml_18, self).__init__()
        h5py_file = h5py.File(dataset_path, 'r')
        self.data = h5py_file['X']
        self.modulations = np.argmax(h5py_file['Y'], axis=1)
        self.snr = h5py_file['Z'][:, 0]
        self.snr_ratio = snr_ratio
        self.sequence_length = sequence_length if sequence_length is not None else self.data.shape[1]

        # Define full list of modulations
        all_modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
                           'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

        # Default modulations if not provided
        if not isinstance(selected_modulations, list):
            # selected_modulations = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', 'AM-SSB-WC', 
            #                         'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
            selected_modulations = all_modulations
        # Ensure `selected_modulations` exists in the full list
        self.mod_classes = [mod for mod in all_modulations if mod in selected_modulations]

        # Get corresponding indices
        mod_indices_to_include = [all_modulations.index(mod) for mod in self.mod_classes]

        self.snr_classes = np.arange(-20., 31., 2)

        # Filter data based on selected modulations
        data_masking = np.isin(self.modulations, mod_indices_to_include)
        self.data = self.data[data_masking]
        self.modulations = self.modulations[data_masking]
        self.snr = self.snr[data_masking]

        # Remap the modulation labels to sequential values
        self.label_mapping = {original_label: new_label for new_label, original_label in enumerate(mod_indices_to_include)}
        self.modulations = np.array([self.label_mapping[mod] for mod in self.modulations])

        np.random.seed(2018)
        train_indices, validation_indices, test_indices = [], [], []

        # Iterate over the selected modulation indices
        for new_mod_label in range(len(mod_indices_to_include)):
            mod_mask = self.modulations == new_mod_label
            mod_indices = np.where(mod_mask)[0]

            for snr_idx in range(0, 26):  # All signal to noise ratios from (-20, 30) dB
                snr_mask = self.snr[mod_indices] == self.snr_classes[snr_idx]
                indices_subclass = mod_indices[snr_mask]

                if len(indices_subclass) == 0:
                    continue

                np.random.shuffle(indices_subclass)
                train_indices_sublcass = indices_subclass[:int(0.7 * len(indices_subclass))]
                validation_indices_subclass = indices_subclass[int(0.7 * len(indices_subclass)):int(0.85 * len(indices_subclass))]
                test_indices_subclass = indices_subclass[int(0.85 * len(indices_subclass)):]

                if snr_idx >= snr_ratio:
                    train_indices.extend(train_indices_sublcass)
                    validation_indices.extend(validation_indices_subclass)
                    test_indices.extend(test_indices_subclass)

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

        print('Filtered dataset shape:', self.data.shape)
        print("Train indices for selected SNRs:", len(train_indices))
        print("Validation indices for selected SNRs:", len(validation_indices))
        print("Test indices for selected SNRs:", len(test_indices))

        input_length = self.data.shape[1]
        print("Input length:", input_length)

    def __getitem__(self, index):
        assert self.sequence_length <= self.data.shape[1], \
            f"Sequence length {self.sequence_length} exceeds data length {self.data.shape[2]}"

        sequence = self.data[index, :self.sequence_length, :]
        label = self.modulations[index]
        return torch.tensor(sequence.transpose(), dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(self.snr[index])

    def __len__(self): 
        return self.data.shape[0]

    def get_original_label(self, new_label):
        """Get the original modulation label from the new label."""
        for orig_label, mapped_label in self.label_mapping.items():
            if mapped_label == new_label:
                return self.mod_classes[orig_label]
            

def quantize(data):
    quant_min = -2.0
    quant_max = 2.0
    quant_range = quant_max - quant_min
    data_quant = (data - quant_min) / quant_range
    data_quant = np.round(data_quant * 256) - 128
    data_quant = np.clip(data_quant, -128, 127)
    data_quant = data_quant.astype(np.int8)
    return data_quant
# def quantize(data):
#     quant_min = -2.0
#     quant_max = 2.0
#     quant_range = quant_max - quant_min  # 4.0
#     data_normalized = (data - quant_min) / quant_range
#     data_quant = np.floor(data_normalized * 4 + 0.5)  # round to nearest
#     data_quant = np.clip(data_quant, 0, 3).astype(np.uint8)
#     return data_quant


model = ModelWrapper("./export/wise-lion-17/text_export_qonnx.onnx")
model = cleanup_model(model)
batch_size = 1
num_batches = 1024


dataset = Radioml_18(dataset_path, selected_modulations=['BPSK', 'QPSK'], sequence_length=128, snr_ratio=25)
data_loader_test_overall = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
mod_classes = dataset.mod_classes
snr_classes = dataset.snr_classes



def quantize(data):
    quant_min = -2.0
    quant_max = 2.0
    quant_range = quant_max - quant_min
    data_quant = (data - quant_min) / quant_range
    data_quant = np.round(data_quant * 256) - 128
    data_quant = np.clip(data_quant, -128, 127)
    data_quant = data_quant.astype(np.int8)
    return data_quant



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
