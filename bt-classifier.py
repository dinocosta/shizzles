"""
Neuronal network capable of classifying if brain tissue can be seen on a 
given MRI image.
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
DS_PATH = '/mnt/disk3/datasets_rm/data_set_skull'
CLASSES_PATH = '/mnt/disk3/datasets_rm/data_set_skull/dl_skull_trab/classes/'

# Array of exams and array of respective classifications.
mris    = []
classes = []

def to_binary_vector(value):
    if value == 1:
        return [0.0, 1.0]
    else:
        return [1.0, 0.0]

files_list = sorted(os.listdir(DS_PATH))
for i in range(len(files_list)):
    filepath = DS_PATH + '/' + files_list[i]
    if ('mask' not in filepath):
        mris.append(nib.load(filepath))
        classes.append(np.loadtxt(CLASSES_PATH + 'class_' + files_list[i] + '.txt', dtype=int))

mris = list(map(lambda x: np.rollaxis(x.get_data(), 2), mris))

# Divide 80% of the dataset for training and 20% for evaluation.
ds_size         = len(mris)
train_size      = int(np.floor(ds_size * 0.8))
train_mris      = mris[:train_size]
train_classes   = classes[:train_size]
test_mris       = mris[train_size:]
test_classes    = classes[train_size:]

# Convert class vectors to binary class matrices.
train_classes   = list(map(lambda x: list(map(lambda y: to_binary_vector(y), x)), train_classes))
test_classes    = list(map(lambda x: list(map(lambda y: to_binary_vector(y), x)), test_classes))

# Create model.
model = Sequential()
model.add(Conv2D(32, activation='relu', input_shape=(mris[0].shape[1], mris[0].shape[2]), nb_row=255, nb_col=176))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizer.Adadelta(),
        metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(test_mris, test_classes))
score = model.evaluate(test_mris, test_classes, verbose=0)
print('Test loss: ', score[0])
# print('Test accuracy: ', score[1])
