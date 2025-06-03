
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchsummary import summary
from sklearn.metrics import accuracy_score
import h5py
import pandas as pd
import plotly.express as px
import numpy as np
import wandb
from pathlib import Path
import models

### CONFIGURATION ###

# Select which GPU to use (if available)
gpu = 0

config_base = {
  "batch_size" : 1024,
  "epochs" : 1,
  "lr": 0.01,
  "lr_scheduler": "EXP", # EXP or CAWR
  "lr_scheduler_halflife": 10, # only for EXP: half life in epochs
  "lr_scheduler_t0": 5, # only for CAWR: t_0
  "lr_scheduler_tmult": 1, # only for CAWR: t_mult
  "train_snr_cutoff": 30, # do not train on data below this SNR (in dB)
}

config_vgg10_float = dict(config_base)
config_vgg10_float.update({
  "base_topology": "VGG10_float_v1",
  "filters_conv": 80,
  "filters_dense": 128,
  "dropout_conv" : 0.0,
  "dropout_dense": 0.0,
})

config_vgg24_float = dict(config_base)
config_vgg24_float.update({
  "base_topology": "VGG24_float_v1",
  "filters_conv": 40,
  "filters_dense": 128,
  "dropout_conv" : 0.0,
  "dropout_dense": 0.0,
})

config_vgg10_quant = dict(config_base)
config_vgg10_quant.update({
  "base_topology": "VGG10_quant_v1",
  "filters_conv": 40,
  "filters_dense": 128,
  "dropout_conv" : 0.0,
  "dropout_dense": 0.0,
  "in_quant_range": 2.0,
  "in_quant_bits": 8,
  "a_bits_l1": 4, #first layer
  "w_bits_l1": 4, #first layer
  "a_bits": 4,
  "w_bits": 4,
  "a_quant_type": "Linear", # ReLU or Linear
  "a_quant_min": -1.0, # only for Linear
  "a_quant_max": 1.0, # only for Linear
  "w_quant_channelwise_scaling": False,
})

config_vgg24_quant = dict(config_base)
config_vgg24_quant.update({
  "base_topology": "VGG24_quant_v1",
  "filters_conv": 24,
  "filters_dense": 128,
  "dropout_conv" : 0.0,
  "dropout_dense": 0.0,
  "in_quant_range": 2.0,
  "in_quant_bits": 8,
  "a_bits_l1": 4, #first layer
  "w_bits_l1": 4, #first layer
  "a_bits": 4,
  "w_bits": 4,
  "a_quant_type": "Linear", # ReLU or Linear
  "a_quant_min": -1.0, # only for Linear
  "a_quant_max": 1.0, # only for Linear
  "w_quant_channelwise_scaling": False,
})

config_bacalhau = dict(config_base)
config_bacalhau.update({
  "base_topology": "Bacalhaunet_v1",
  "filters_conv1": 99,
  "filters_conv2": 99,
  "a_bits": 4,
  "w_bits": 4,
})

config_bacalhau_float = dict(config_base)
config_bacalhau_float.update({
  "base_topology": "Bacalhaunet_v1_float",
  "filters_conv1": 24,
  "filters_conv2": 80
})

wandb.init(project="RadioML", entity="felixj",
           tags=[],
           config=config_base,
          )

### CONFIGURATION END ###

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:' + str(gpu))
    print("Using GPU %d" % gpu)
else:
    device = torch.device('cpu')
    print("Using CPU only")

# Setting seeds for reproducibility
torch.manual_seed(123456)
np.random.seed(123456)

# load base topology from models.py
model = getattr(models, wandb.config.base_topology)(wandb.config)

# manual weight initialization
#def init_weights(m):
#    if type(m) == nn.BatchNorm1d:
#        # manually initialize weights to 1 and biases to 0
#        m.reset_parameters()
#model.apply(init_weights)

#x = torch.randn(1, 2, 1024).cpu()
#summary(model.cuda(), (2,1024))

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

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
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
                
                snr_index_cutoff = (wandb.config.train_snr_cutoff+20)/2
                if snr_idx >= snr_index_cutoff:
                    train_indices.extend(train_indices_subclass)
                    # validation data = test data from same SNR range as training data
                    val_indices.extend(test_indices_subclass)

                test_indices_overall.extend(test_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        self.test_sampler_overall = torch.utils.data.SubsetRandomSampler(test_indices_overall)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

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
    for mod in range(24):
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

dataset = radioml_18_dataset(dataset_path)
data_loader_train = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=dataset.train_sampler)
data_loader_val = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=dataset.val_sampler)
data_loader_test_overall = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=dataset.test_sampler_overall)

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

# log torch model topology, params, gradients
#watch(
#    models, criterion=None, log="gradients", log_freq=1000, idx=None,
#    log_graph=(False)
#)
