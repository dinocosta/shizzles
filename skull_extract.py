import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
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

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.rollaxis(xt.get_data().reshape((1,) + xt.shape + (1,)), 3, 1)
X_test = np.rollaxis(xtt.get_data().reshape((1,) + xtt.shape + (1,)), 3, 1)

Y_train = np.rollaxis(yt.get_data(), 2, 0)
Y_test = np.rollaxis(ytt.get_data(), 2, 0)

print(X_train.dtype)
print(np.amax(X_train))
print(np.amax(X_test))
print("X_train: ", X_train)
print("X_train[0]: ", X_train[0])
print("X_train[0][0]: ", X_train[0][0])
print("len X_train: ", len(X_train))
print("len X_train[0]: ", len(X_train[0]))
print("len X_train[0][0]: ", len(X_train[0][0]))


print("Y_train: ", Y_train)
print("Y_train[0]: ", Y_train[0])
print("Y_train[0][0]: ", Y_train[0][0])
print("len Y_train: ", len(Y_train))
print("len Y_train[0]: ", len(Y_train[0]))
print("len Y_train[0][0]: ", len(Y_train[0][0]))

model = Sequential()
# First layer. We need to tell it the input_shape(rows,cols,channels)
model.add(Convolution3D(32,3,3,3, activation="relu", input_shape=(200, 176, 251, 1)))
# Second layer
model.add(Convolution3D(32, 3, 3, 3, activation='relu'))
# Third layer - MaxPooling. Size decreases here
model.add(MaxPooling3D(pool_size=(2,2,2)))
# Dropout is a technique to reduce overfitting probabilty
model.add(Dropout(0.25))
# Weights need to be flattened before the fully connected layer (dense)
model.add(Flatten())
# Fully Connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Output layer
model.add(Dense(10, activation='linear'))

# Declare the loss function and the optimizer
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
