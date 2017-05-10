"""
bt-classifier: Brain Tissue Classifier

Neuronal network capable of classifying if brain tissue can be seen on a 
given MRI image.

There are two classes:
    0 - No brain tissue can be seen on the MRI image.
    1 - Brain tissue can be seen on the MRI image.
"""

import numpy as np
import nibabel as nib
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size  = 128
num_classes = 2
epochs      = 1

# Dataset path.
MRIS_PATH = '/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/mris'
CLASSES_PATH = '/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/classes/'

# Array of exams and array of respective classifications.
mris    = []
classes = []

def to_binary_vector(value):
    """
    Helper function which converts binary values to class vector.

    Arguments:
        value - The binary value
    Returns:
        Array with the corresponding class
    """
    if value == 1:
        return [0.0, 1.0]
    else: 
        return [1.0, 0.0]

# Load mris and classes
print("Loading MRIs and classes...")
files_list = sorted(os.listdir(MRIS_PATH))
for i in range(len(files_list)):
    filepath = MRIS_PATH + '/' + files_list[i]
    mris.append(nib.load(filepath).get_data())
    classes.append(np.loadtxt(
        fname = CLASSES_PATH + 'class_' + files_list[i] + '.txt', 
        dtype = int))

# Concatenate, roll and reshape mris so shape becames (exams, width, height, depth)
mris = np.concatenate(mris, 2)
mris = np.rollaxis(mris, 2)
mris = mris.reshape(mris.shape[0], 176, 256, 1)

# Convert class vectors to binary class matrices and concatenate classes
classes = np.array(list(map(
    lambda x: list(map(lambda y: to_binary_vector(y), x)), 
    classes)))
classes = np.concatenate(classes, 0)

# Divide 80% of the dataset for training and 20% for evaluation.
print("Dividing dataset...")

ds_size         = len(mris)
train_size      = int(np.floor(ds_size * 0.8))
train_mris      = mris[:train_size]
train_classes   = classes[:train_size]
test_mris       = mris[train_size:]
test_classes    = classes[train_size:]

# Confirm shapes
print('x_train shape: ', train_mris.shape)
print('y_train shape: ', train_classes.shape)
print('x_test shape: ', test_mris.shape)
print('y_test shape: ', test_classes.shape)

# Create model.
print("Creating model...")

model = Sequential()
model.add(Conv2D(
    filters     = 32, 
    kernel_size = (3, 3), 
    activation  = 'relu',
    input_shape = (176, 256, 1)))
model.add(Conv2D(
    filters     = 64, 
    kernel_size = (3, 3), 
    activation  = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(
    rate = 0.25))
model.add(Dense(
    units       = 128, 
    activation  = 'relu'))
model.add(Dropout(
    rate = 0.5))
model.add(Dense(
    units       = num_classes, 
    activation  = 'softmax'))

model.compile(
    loss        = 'categorical_crossentropy',
    optimizer   = 'adadelta',
    metrics     = ['accuracy'])

model.fit(
    train_mris, 
    train_classes,
    batch_size      = batch_size,
    epochs          = epochs,
    verbose         = 1,
    validation_data = (test_mris, test_classes))
score = model.evaluate(
    x       = test_mris, 
    y       = test_classes, 
    verbose = 0)

print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
