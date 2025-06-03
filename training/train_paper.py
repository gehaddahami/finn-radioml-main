
import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn
from torchsummary import summary
from sklearn.metrics import accuracy_score
import h5py
import pandas as pd
import plotly.express as px
import numpy as np
import wandb
from pathlib import Path
import models_test

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

def train(model, train_loader, optimizer, criterion):
    # ensure model is in training mode
    model.train()

    losses = []
    for (inputs, target, snr) in train_loader:   
        inputs = inputs.to(device)
        target = target.to(device)
                
        # forward pass
        output = model(inputs)
        loss = criterion(output, target)
        
        # backward pass + run optimizer to update weights
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())
           
    return np.mean(losses)

def test_loss(model, test_loader, criterion):    
    # ensure model is in eval mode
    model.eval() 

    losses = []
    with torch.no_grad():
        for (inputs, target, snr) in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)

            loss = criterion(output, target)
            losses.append(loss.cpu().detach().numpy())
        
    return np.mean(losses)

def test_accuracy(model, test_loader, mod_classes, snr_classes):    
    # ensure model is in eval mode
    model.eval() 

    # inference
    y_exp = np.empty((0))
    y_snr = np.empty((0))
    y_pred = np.empty((0,len(mod_classes)))
    with torch.no_grad():
        for (input, target, snr) in test_loader:
            input = input.to(device)

            output = model(input)

            y_pred = np.concatenate((y_pred,output.cpu()))
            y_exp = np.concatenate((y_exp,target))
            y_snr = np.concatenate((y_snr,snr))

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

    confusion_matrix_overall = px.imshow(confnorm_overall, x=mod_classes, y=mod_classes, color_continuous_scale="blues",
        labels={'x': "Predicted Class", 'y': "True Class", 'color': "prob."})
    confusion_matrix_30dB = px.imshow(confnorm[-1,:,:], x=mod_classes, y=mod_classes, color_continuous_scale="blues",
        labels={'x': "Predicted Class", 'y': "True Class", 'color': "prob."})
    accuracy_overall = np.sum(np.diag(conf_overall)) / len(y_exp)
    accuracy_30dB = accuracy_per_SNR[-1]
    
    # accuracy-over-SNR plot over all modulations
    accuracy_plot_overall = px.line(x=dataset.snr_classes, y=accuracy_per_SNR, labels={'x': "SNR [dB]", 'y': "Accuracy"}, markers=True)

    # accuracy-over-SNR plot per modulation
    data = []
    for mod in range(len(mod_classes)):
        for snr in snr_classes:
            indices = ((y_exp == mod) & (y_snr == snr)).nonzero()
            y_exp_i = y_exp[indices]
            y_pred_i = y_pred[indices]
            cor = np.count_nonzero(y_exp_i == np.argmax(y_pred_i, axis=1))
            data.append([mod_classes[mod], snr, cor/len(y_exp_i)])

    df = pd.DataFrame(data, columns=["Modulation", "SNR [dB]", "Accuracy"])
    accuracy_plot_per_mod = px.line(df, x="SNR [dB]", y="Accuracy", color="Modulation", markers=True)

    plots = {"confusion_matrix_overall": confusion_matrix_overall, 
             "confusion_matrix_30dB": confusion_matrix_30dB, 
             "test_acc_plot_overall": accuracy_plot_overall, 
             "test_acc_plot": accuracy_plot_per_mod}
    return accuracy_overall, accuracy_30dB, plots

if 'DATASET_PATH_RADIOML' in os.environ:
    dataset_path = os.environ['DATASET_PATH_RADIOML'] + "/2018/GOLD_XYZ_OSC.0001_1024.hdf5"
    if not os.path.isfile(dataset_path):
        print("ERROR: Dataset not found")
else:
    dataset_path = '/home/gehad/Downloads/dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'


# Select which GPU to use (if available)
gpu = 0

config_vgg7_float = {
  "base_topology": "VGG7_float_v1",
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
  "dropout_conv" : 0.0,
  "dropout_dense": 0.0,
}

config_vgg7_quant = {
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

configs = [config_vgg7_quant]

for config in configs:
    print(wandb.config)
    wandb.init(project="RadioML_Finn_updated", entity="dahami-gehad-paderborn-university", reinit= True, config=config)

    print('Running: ', wandb.config.base_topology)


    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu))
        print("Using GPU %d" % gpu)
    else:
        device = torch.device('cpu')
        print("Using CPU only")

    # load dataset
    dataset = Radioml_18(dataset_path, selected_modulations=['BPSK', 'QPSK'], sequence_length=128)
    data_loader_train = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=dataset.train_sampler)
    data_loader_val = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=dataset.validation_sampler)
    data_loader_test_overall = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=dataset.test_sampler)

    # Setting seeds for reproducibility
    torch.manual_seed(2025)
    np.random.seed(2025)

    # load base topology from models.py
    model = getattr(models_test, wandb.config.base_topology)(wandb.config)

    model = model.to(device)

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

    if (wandb.config.lr_scheduler == "EXP"):
        gamma = 0.5**(1/wandb.config.lr_scheduler_halflife)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif (wandb.config.lr_scheduler == "CAWR"):
        t_0 = wandb.config.lr_scheduler_t0
        t_mult = wandb.config.lr_scheduler_tmult
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    else:
        print("ERROR: Unknown lr_scheduler specified")

    # prepare save directory for this run
    save_path = "trained_models/run_" + wandb.run.name + "_" + wandb.run.id
    Path(save_path).mkdir(parents=True, exist_ok=True)

    best_acc_overall = 0
    best_acc_30dB = 0
    best_plots = None

    for epoch in range(wandb.config.epochs):
        train_loss = train(model, data_loader_train, optimizer, criterion)
        val_loss = test_loss(model, data_loader_val, criterion)
        test_acc_overall, test_acc_30dB, plots = test_accuracy(model, data_loader_test_overall, dataset.mod_classes, dataset.snr_classes)

        print("Epoch %d: test_acc_overall = %f, test_acc_30dB = %f" % (epoch, test_acc_overall, test_acc_30dB))
        lr_scheduler.step()

        # save (and overwrite) best model (according to 30 dB test acc only)
        if test_acc_30dB > best_acc_30dB:
            best_acc_overall = test_acc_overall
            best_acc_30dB = test_acc_30dB
            best_plots = plots
            optimizer_state = {'epoch': epoch, 'optimizer': optimizer.state_dict()}
            torch.save(optimizer_state, save_path + "/optimizer_state.pth")
            torch.save(model.state_dict(), save_path + "/model_state.pth")
            #torch.save(model, save_path + "/saved_model") # redundant, save just in case, does not work with brevitas (Can't pickle <class 'brevitas.inject.NoneActQuant'>)!
        
        log_dict = {"train_loss": train_loss, "val_loss": val_loss, "test_acc_overall": test_acc_overall, "test_acc_30dB": test_acc_30dB}
        # log plots only after last epoch
        if epoch == wandb.config.epochs - 1:
            log_dict.update(best_plots)
        wandb.log(log_dict)

        # update summary metrics (otherwise these will be set to the last logged step)
        wandb.summary["test_acc_overall"] = best_acc_overall
        wandb.summary["test_acc_30dB"] = best_acc_30dB

    # log final (best) model to wandb
    artifact = wandb.Artifact(wandb.config.base_topology, type='trained_model')
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)


    print('Done: ', wandb.config.base_topology)
    print('--------------------------------')
    wandb.finish()

# log torch model topology, params, gradients
#watch(
#    models, criterion=None, log="gradients", log_freq=1000, idx=None,
#    log_graph=(False)
#)
