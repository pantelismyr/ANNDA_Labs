'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import keras.utils as keras_utils
import numpy as np
import csv


# ------------------------ DATA COLLECTION -----------------------------

def getData(filename):
    with open(filename, 'r') as f:
        # read csv file
        rows = csv.reader(f)
        # create a list of lists
        ls = list(rows)
        # convert strings to integers
        ls = [[int(i) for i in row] for row in ls]

    return ls

def get_datasets():
    x_train = getData("binMNIST_data/bindigit_trn.csv")
    x_test = getData("binMNIST_data/bindigit_tst.csv")
    y_train = getData("binMNIST_data/targetdigit_trn.csv")
    y_test = getData("binMNIST_data/targetdigit_tst.csv")

    return np.asarray(x_train), np.asarray(x_test), \
           keras_utils.to_categorical(y_train, 10), keras_utils.to_categorical(y_test, 10)


# ----------------------- WEIGHT GENERATOR -------------------------------

def generate_weight_matrix(number_of_inputs, number_of_nodes):

    weight_matrix = np.ndarray(shape=(number_of_inputs, number_of_nodes))

    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):

            weight_matrix[i,j] = np.random.normal(0,0.01)

    return weight_matrix

batch_size = 128
num_classes = 10
epochs = 100

# the data, split between train and test sets
x_train, x_test, y_train, y_test = get_datasets()

# convert class vectors to binary class matrices

model = Sequential()
model.add(Dense(150, activation='relu', input_shape=(784,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])