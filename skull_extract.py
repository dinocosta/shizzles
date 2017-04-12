import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import normalization as norm

# NIFTI
import os
import nibabel as nib
from nibabel.testing import data_path

mris = []
masks = []
# Load
files_list = sorted(os.listdir("/home/dl_skull/normalized_images"))
# for i in range(len(files_list)):
for i in range(50):
    file = "/home/dl_skull/normalized_images/" + files_list[i]
    file = nib.load(file)
    file = file.get_data()
    if (i%2==0):
        mris.append(file)
    else:
        masks.append(file)

xt = np.concatenate(mris[0:15], 2)
print(xt.shape)
yt = np.concatenate(masks[0:15], 2)
print(yt.shape)
xtt = np.concatenate(mris[15:], 2)
print(xtt.shape)
ytt = np.concatenate(masks[15:], 2)
print(ytt.shape)

X_train = np.rollaxis(norm.normalize_image(xt), 2).reshape(2967, 176, 256, 1)
Y_train = np.rollaxis(norm.normalize_image(yt), 2).reshape(2967, 176, 256, 1)
X_test = np.rollaxis(norm.normalize_image(xtt), 2).reshape(1945, 176, 256, 1)
Y_test = np.rollaxis(norm.normalize_image(ytt), 2).reshape(1945, 176, 256, 1)

model = Sequential()
model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='default', input_shape=(176, 256, 1)))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dense(1))
#model.add(Dropout(0.2))
#model.add(Dense(10000))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
#model.add(Dense(44176))
# 
 
# Declare the loss function and the optimizer
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
 
# Train model
model.fit(X_train, Y_train, batch_size=3, nb_epoch=1, verbose=1)
 
# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
# print(score)
