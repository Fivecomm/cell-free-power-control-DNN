#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script consists of the preprocessing part of the research work:

G. García-Barrios, M. Fuentes and D, Martín-Sacristán, "Data-Driven Energy 
Efficiency Modeling in Large-Scale Networks: An Expert Knowledge and ML-Based 
Approach," in IEEE Transactions on Machine Learning in Communications and 
Networking. [Submitted]

This is version 1.0 (Last edited: 2024-10-18)

@author: Guillermo Garcia-Barrios

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file.
"""

from tensorflow import keras
import hdf5storage
import numpy as np
import os
import time

##############################################################################
## PARAMETERS TO SET UP

# Cell-free scenario
SCN_TRAIN = 'scenario_07'
SCN_TEST = 'scenario_05'

# Power control strategy: 'maxmin', 'sumSE', 'FPC'
PC_STRATEGY = 'sumSE'

##############################################################################
## LOAD DATA

# Load scenario data
mat_contents = hdf5storage.loadmat('data/' + SCN_TRAIN + '.mat')
K = int(mat_contents['K'])   # number of UEs
L = int(mat_contents['L'])   # number of APs

# Load test data
betas = np.load('data/' + SCN_TEST + '/betas_' + PC_STRATEGY + '_test.npy')
nbr_setups_test = len(betas)
K_test = len(betas[0])

if K_test < K:
    # Rearrange input data fitting with very small values the missing UEs
    small_value = 10 ** -20
    input = small_value * np.ones((nbr_setups_test, K))
    input[:,0:K_test] = betas
else:
    input = betas

# Load model
model = keras.models.load_model('models/' + SCN_TRAIN + '/' + PC_STRATEGY + 
                                '.keras')

##############################################################################
## TEST MODEL

start_time = time.time()

output = model.predict(input)

elapsed_time = time.time() - start_time
print('Test time:', elapsed_time, 'seconds')

##############################################################################
## SAVE DATA

# Create directory it doesn't exist
saving_path = 'results/train_' + SCN_TRAIN + '-test_' + SCN_TEST + '/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

np.save(saving_path + 'pk_' + PC_STRATEGY + '_output.npy',output)   