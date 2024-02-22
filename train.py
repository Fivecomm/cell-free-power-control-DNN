#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script consists of the preprocessing part of the research work:

Guillermo García-Barrios, Manuel Fuentes, David Martín-Sacristán, "A Flexible
Low-Complexity DNN Solution for Power Control in Cell-Free Massive MIMO," 2024
IEEE Annual International Symposium on Personal, Indoor, and Mobile Radio
Communications (IEEE PIMRC 2024), Valencia, Spain, 2024. [Pending acceptance]

This is version 1.0 (Last edited: 2024-02-05)

@author: Guillermo Garcia-Barrios

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file.
"""

import numpy as np
import hdf5storage
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import optimizers
import time
import os
import matplotlib.pyplot as plt

##############################################################################
## PARAMETERS TO SET UP

# Data path
DATA_PATH = 'dataset/'

# Cell-free scenario
SCENARIO = 'scenario_01'

# Power control strategy: 'maxmin', 'sumSE', 'FPC'
PC_STRATEGY = 'FPC'

##############################################################################
## LOAD DATA

input = np.load('data/' + SCENARIO + '/betas_' + PC_STRATEGY + '_train.npy')
output = np.load('data/' + SCENARIO + '/pk_' + PC_STRATEGY + '_train.npy')

# Number of input parameters to optimize
N_inputs = len(input[0])

# Load scenario data
mat_contents = hdf5storage.loadmat(DATA_PATH + SCENARIO + '/setup.mat')
K = int(mat_contents['K'])   # number of UEs
L = int(mat_contents['L'])   # number of APs

##############################################################################
## DEFINE MODEL

# Maximum number of epochs
N_max_epoch = 100
# Batch size
N_batch_size = 128

# Setting the number of layers and number of neurons per layer
N_L = np.array([128, 64, K])

# Neural network configuration
model = Sequential()
model.add(Dense(N_L[0], activation='relu', input_shape=(N_inputs,),
                name ='layer0'))
model.add(Dropout(0.5))
model.add(Dense(N_L[1], activation='relu',name ='layer1'))
model.add(Dense(N_L[2], activation='sigmoid',name ='layer2'))

print(model.summary())

##############################################################################
## TRAIN MODEL

start_time = time.time()

# Initialize the weights of the model
keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)

# Optimizer
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)

# Training
model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
history1 = model.fit(input, output, validation_split=0.1, epochs=N_max_epoch, 
                    batch_size=N_batch_size, verbose=2)

elapsed_time = time.time() - start_time
print('Training time:', elapsed_time, 'seconds')

##############################################################################
## LOSS FUNCTION

plt.figure(figsize=(10, 5))
plt.plot(history1.history['loss'], label='Training loss')
plt.plot(history1.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history1.history['accuracy'], label='Training accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()

##############################################################################
## SAVE MODEL

# Create directory it doesn't exist
saving_path = 'models/' + SCENARIO + '/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

model.save(saving_path + PC_STRATEGY + '.keras')