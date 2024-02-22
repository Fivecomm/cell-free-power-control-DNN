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

from tensorflow import keras
import numpy as np
import os
import time

##############################################################################
## PARAMETERS TO SET UP

# Cell-free scenario
SCENARIO = 'scenario_01'

# Power control strategy: 'maxmin', 'sumSE', 'FPC'
PC_STRATEGY = 'maxmin'

##############################################################################
## LOAD DATA

# Load test data
input = np.load('data/' + SCENARIO + '/betas_' + PC_STRATEGY + '_test.npy')
# Load model
model = keras.models.load_model('models/' + SCENARIO + '/' + PC_STRATEGY + 
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
saving_path = 'results/' + SCENARIO + '/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

np.save(saving_path + 'pk_' + PC_STRATEGY + '_output.npy',output)   