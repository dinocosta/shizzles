import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding3D
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

# NIFTI
import os
import nibabel as nib
from nibabel.testing import data_path
# Load
X_train = '/home/dl_skull/ex1.nii.gz'
Y_train = '/home/dl_skull/ex1-res.nii.gz'
X_test = '/home/dl_skull/ex2.nii.gz'
Y_test = '/home/dl_skull/ex2-res.nii.gz'

[xt, yt, xtt, ytt] = map(nib.load, [X_train, Y_train, X_test, Y_test])

print(xt.shape)
print(yt.shape)
print(xtt.shape)
print(ytt.shape)

X_train = np.rollaxis(xt.get_data(), 2).reshape(200, 44176)
Y_train = np.rollaxis(yt.get_data(), 2).reshape(200, 44176)
 
model = Sequential()
model.add(Dense(5000, input_shape=(44176,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10000))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(44176))
# model.add(ZeroPadding3D(padding=(1, 1, 1), dim_ordering='default', input_shape=(200, 176, 251, 1)))
# 
# model.add(Convolution3D(1, 3, 3, 3, activation="relu" ))
 
# Declare the loss function and the optimizer
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
 
# Train model
model.fit(X_train, Y_train, batch_size=3, nb_epoch=1, verbose=1)
# 
# # Evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
