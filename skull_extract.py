import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras_diagram import ascii
from matplotlib import pyplot as plt
import time
import sys
import psutil

# NIFTI
import os
import nibabel as nib
from nibabel.testing import data_path

mris = []
masks = []
# Load images, matching by alphabetical order
mris_list = sorted(os.listdir("/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/mris"))
masks_list = sorted(os.listdir("/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/masks"))

print("Loading MRI's...")
for i in range(len(mris_list)):
    filepath = "/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/mris/" + mris_list[i]
    file = nib.load(filepath).get_data()
    mris.append(file)

print("Loading masks...")
for i in range(len(mris_list)):
    filepath = "/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/masks/" + masks_list[i]
    file = nib.load(filepath).get_data()
    masks.append(file)


# Concatenate exams on the samples axis
mris = np.concatenate(mris, 2)
masks = np.concatenate(masks, 2)

# Order the dimensions to have (samples, rows, cols, channels)
mris = np.rollaxis(mris, 2).reshape(mris.shape[2], 176, 256, 1)
masks = np.rollaxis(masks, 2).reshape(masks.shape[2], 176, 256, 1)

# Remove exams that don't have brain tissue. Training only for those who do.
no_brain_list = []
for i in range(len(masks)):
    if (np.max(masks[i]) == 0):
        no_brain_list.append(i)
masks = np.delete(masks, no_brain_list, axis=0)
mris = np.delete(mris, no_brain_list, axis=0)

# Split data for training and
split_len = int(len(mris) * 0.8)

mris_training = mris[:split_len]
masks_training = masks[:split_len]
mris_evaluate = mris[split_len:]
masks_evaluate = masks[split_len:]

model = Sequential()
model.add(ZeroPadding2D(padding=(3, 3), input_shape=(176, 256, 1)))
model.add(Convolution2D(64, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
 
# Declare the loss function and the optimizer
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
 
# Train model
history = model.fit(mris_training, masks_training, batch_size=8, epochs=10, verbose=1, validation_split=0.2)
p = psutil.Process()
cpu_time = p.cpu_times()[0]
mem = p.memory_info()[0]

# Save Model
timestamp = time.strftime("%Y%m%d-%H%M%S")
model.save("logs/"+ timestamp + ".h5")

# Evaluate model
print("###########Evaluation############\n")
scores = model.evaluate(mris_evaluate, masks_evaluate)
print("\n Metric: %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))

# Log results
with open("logs/log.csv", "a") as myfile:
    myfile.write(",")
    myfile.write("\"" + ascii(model).strip() + "\",")
    myfile.write("{},".format(history.history['loss'][-1]))
    myfile.write("{},".format(history.history['acc'][-1]))
    myfile.write("{},".format(mem))
    myfile.write("{},\n".format(cpu_time))
