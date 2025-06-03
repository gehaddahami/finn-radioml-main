import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
import operator

### DEFINE DATALOADER ###
class radioml_18_dataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, train, batch_size, small=False):
        super(radioml_18_dataset, self).__init__()
        self.batch_size = batch_size

        h5_file = h5py.File(data_dir + "/2018/GOLD_XYZ_OSC.0001_1024.hdf5",'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]

        #self.len = self.data.shape[0]

        # note: labels given in the "classes.txt" file are not in the correct order (https://github.com/radioML/dataset/issues/25)
        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']

        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # data partitioning via samplers
        np.random.seed(2018)
        train_indices = []
        val_indices = []
        for mod in range(0, 24): #all modulations (0 to 23)
            for snr_idx in range(0, 26): #all SNRs (0 to 25 = -20dB to +30dB)
                start_idx = 26*4096*mod + 4096*snr_idx
                indices_subclass = list(range(start_idx, start_idx+4096))
                
                split = int(np.ceil(0.1 * 4096)) #90%/10% split
                np.random.shuffle(indices_subclass)
                train_indices_subclass, val_indices_subclass = indices_subclass[split:], indices_subclass[:split]

                if snr_idx >= 7: #only SNR >= -6dB for training
                    train_indices.extend(train_indices_subclass)
                val_indices.extend(val_indices_subclass)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        if(train):
            self.indices = train_indices
        else:
            self.indices = val_indices

        if(small):
            self.indices = self.indices[:1000]

        self.len = len(self.indices)

    def __getitem__(self, idx):
        #f = operator.itemgetter(self.indices[idx*self.batch_size:(idx+1)*self.batch_size])
        #return np.array(f(self.data)), np.array(f(self.mod))

        data_batch = []
        label_batch = []
        for i in range(self.batch_size):
            data_batch.append(self.data[self.indices[idx*self.batch_size + i]])
            label_batch.append(self.mod[self.indices[idx*self.batch_size + i]])
        
        return np.array(data_batch).reshape((-1,1,1024,2)), np.array(label_batch)

    def __len__(self):
        return int(self.len / self.batch_size)

    def generator(self):
        i = 0
        while i<self.len:
            #yield np.array(self.data[self.indices[i]]).reshape(1, 1024, 2) #first array dimension maps to inputs during calibration
            yield [np.array(self.data[self.indices[i]]).reshape((-1, 1, 1024, 2))]
            i += 1

### DEFINE MODEL ###
filters_conv = 64
filters_dense = 128
model = keras.Sequential(
            [
                #keras.layers.Reshape((1, 1024, 2), input_shape=(1024, 2)),
                keras.layers.Conv2D(filters_conv, (1,3), input_shape=(1, 1024, 2)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Conv2D(filters_conv, (1,3)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Conv2D(filters_conv, (1,3)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Conv2D(filters_conv, (1,3)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Conv2D(filters_conv, (1,3)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Conv2D(filters_conv, (1,3)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Conv2D(filters_conv, (1,3)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),
                keras.layers.MaxPooling2D((1,2)),

                keras.layers.Flatten(),

                keras.layers.Dense(filters_dense),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),

                keras.layers.Dense(filters_dense),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(keras.activations.relu),

                keras.layers.Dense(24),
            ]
        )

print(model.summary())

        
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

train_data = radioml_18_dataset("/workspace/datasets/RadioML", True, 128)
#model.fit(x=train_data, batch_size=128, epochs=1)

# save trained model
tf.saved_model.save(model, "model_trained")

print("Trained model saved, evaluating on test set..")
val_data = radioml_18_dataset("/workspace/datasets/RadioML", False, 128)
#model.evaluate(x=val_data, batch_size=128)

print("Converting model..")
# Convert the model.
representative_dataset = radioml_18_dataset("/workspace/datasets/RadioML", True, 1, small=True).generator

converter = tf.lite.TFLiteConverter.from_saved_model("model_trained")
#converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

# Save the model.
with open('model_quant.tflite', 'wb') as f:
  f.write(tflite_quant_model)

print("Done")
