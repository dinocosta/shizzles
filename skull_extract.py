import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras.datasets import mnist
from keras_diagram import ascii
from keras import backend
from matplotlib import pyplot as plt
import time
import normalization as norm
import sys
import psutil

# NIFTI
import os
import nibabel as nib
from nibabel.testing import data_path


mris = []
masks = []
# Load images
# Splits mris from their masks by ordering and filtering odd from even
files_list = sorted(os.listdir("/home/dl_skull/normalized_images"))
#for i in range(len(files_list)):
for i in range(2):
    filepath = "/home/dl_skull/normalized_images/" + files_list[i]
    file = nib.load(filepath).get_data()
    # Masks are named _bet_mask so they are ordered first
    if ('mask' in filepath):
        masks.append(file)
    else:
        mris.append(file)

# Concatenate exams on the samples axis
mris = np.concatenate(mris, 2)
masks = np.concatenate(masks, 2)

# Order the dimensions to have (samples, rows, cols, channels)
#mris = np.rollaxis(norm.normalize_image(mris), 2).reshape(mris.shape[2], 176, 256, 1)
#masks = np.rollaxis(norm.normalize_image(masks), 2).reshape(masks.shape[2], 176, 256, 1)
mris = np.rollaxis(mris, 2).reshape(mris.shape[2], 176, 256, 1)
masks = np.rollaxis(masks, 2).reshape(masks.shape[2], 176, 256, 1)

model = Sequential()
model.add(ZeroPadding2D(padding=(2, 2), input_shape=(176, 256, 1)))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(1))
 
# Declare the loss function and the optimizer
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
 
# Train model
#history = model.fit(mris, masks, batch_size=3, nb_epoch=5, verbose=1, validation_split=0.2)
history = model.fit(mris, masks, batch_size=3, nb_epoch=1, verbose=1, validation_split=0.2)

p = psutil.Process()
cpu_time = p.cpu_times()[0]
mem = p.memory_info()[0]

# Save Model
timestamp = time.strftime("%Y%m%d-%H%M%S")
model.save("logs/"+ timestamp + ".h5")

# Log results
with open("logs/log.csv", "a") as myfile:
    #myfile.write(timestamp + "\n")
    myfile.write(",")
    myfile.write("\"" + ascii(model).strip() + "\",")
    myfile.write("{},".format(history.history['loss'][-1]))
    myfile.write("{},".format(history.history['acc'][-1]))
    myfile.write("{},".format(mem))
    myfile.write("{},\n".format(cpu_time))
