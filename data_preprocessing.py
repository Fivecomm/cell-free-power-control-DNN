#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script consists of the preprocessing part of the research work:

G. García-Barrios, M. Fuentes, and D. Martín-Sacristán, "A Flexible
Low-Complexity DNN Solution for Power Control in Cell-Free Massive MIMO," 2024
IEEE 35th Annual International Symposium on Personal, Indoor and Mobile Radio 
Communications (PIMRC), Valencia, Spain, 2024.

This is version 1.0 (Last edited: 2024-02-05)

@author: Guillermo Garcia-Barrios

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file.
"""

import hdf5storage
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from pickle import dump

##############################################################################
## PARAMETERS TO SET UP

# Data path
DATA_PATH = 'dataset/'

# Cell-free scenario
SCENARIO = 'scenario_01'

# Power control strategy: 'maxmin', 'sumSE', 'FPC'
PC_STRATEGY = 'maxmin'

# Number of files
NBR_OF_FILES = 20
# Number of setups per file
NBR_OF_SETUPS_PER_FILE = 1000

# Test size
TEST_SIZE = 0.1 # 90% training, 10% testing
# Set fixed value for reproducibility of the test/train split
RANDOM_STATE = 0

##############################################################################
## LOAD DATA

nbr_setups = NBR_OF_FILES * NBR_OF_SETUPS_PER_FILE

# Load scenario data
mat_contents = hdf5storage.loadmat(DATA_PATH + SCENARIO + '/setup.mat')
K = int(mat_contents['K'])  # number of UEs
L = int(mat_contents['L'])  # number of APs
p = mat_contents['p']       # max uplink tx power

# Create variable to store large-scale fading coefficients
betas = np.zeros((L,K,nbr_setups))

# Create variable to store power coefficients
pk = np.zeros((K,nbr_setups))

# Create variable to store SE
SE = np.zeros((K,nbr_setups))

# Variables to calculate SE for the DNN estimations
bk = np.zeros((K,nbr_setups))
ck = np.zeros((K,K,nbr_setups))
sigma2 = np.zeros((K,nbr_setups))

# Name of the power coefficient variable depeding on the optimization scheme
if PC_STRATEGY == 'FPC':
    pk_name = 'pk_UL_nopt_LPMMSE_' + PC_STRATEGY
    ck_name = 'ck_dist_frac'
    bk_name = 'bk_dist_frac'
    sigma2_name = 'sigma2_dist_frac'
else:
    pk_name = 'pk_UL_nopt_LPMMSE_' + PC_STRATEGY
    ck_name = 'ck_dist_full'
    bk_name = 'bk_dist_full'
    sigma2_name = 'sigma2_dist_full'
    
# Join all setups data in one file
for file in range(NBR_OF_FILES):
    
    file_number = '%0*d' % (2, file + 1)
    interval = np.arange(file * NBR_OF_SETUPS_PER_FILE, 
                        (file + 1) * NBR_OF_SETUPS_PER_FILE)  
    
    with h5py.File(DATA_PATH + SCENARIO + '/power_control_part' + 
                   file_number + '.mat', 'r') as f:
        # Large-scale fading coefficients
        data = f.get('gainOverNoise') 
        betas[:,:,interval] = np.transpose(np.array(data))
        # Power coefficients
        data = f.get(pk_name) 
        pk[:,interval] = np.transpose(np.array(data))
        # SE
        data = f.get('SE_UL_nopt_LPMMSE_' + PC_STRATEGY)
        SE[:,interval] = np.transpose(np.array(data))
        # Variables to calculate SE for the DNN estimations
        data = f.get(bk_name)
        bk[:,interval] = np.transpose(np.array(data))
        data = f.get(ck_name)
        ck[:,:,interval] = np.transpose(np.array(data))
        data = f.get(sigma2_name)
        sigma2[:,interval] = np.transpose(np.array(data))        
    
##############################################################################
## PREPROCESSING

# Format to fit with model inputs
betas = np.sum(betas, axis=0)
betas = np.transpose(betas)
pk = np.transpose(pk)

## Based on [Zaher2023]
# Large-scale fading coefficients to dB
betas = 10 * np.log10(betas)
# Power coefficients from mW to dB         
pk = 10 * np.log10(0.001 * pk)

# Standardization of the input features (due to Gaussian distribution)
scaler_input = StandardScaler()
betas = scaler_input.fit_transform(betas)

# Normalization of the output features according to max power
scaler_output = MinMaxScaler()
pk = scaler_output.fit_transform(pk)

##############################################################################
## REPRESENT DATA DISTRIBUTION

# Plot histogram of the inputs
#plt.hist(betas.flatten(), bins=100)
#plt.gca().set(title='Histogram betas', ylabel='Frequency')
#plt.show()

# Plot histogram of the outputs
#plt.hist(pk.flatten(), bins=100)
#plt.gca().set(title='Histogram pk', ylabel='Frequency')
#plt.show()

##############################################################################
## SEPARATE INTO TRAIN AND TEST SPLITS

df_betas = pd.DataFrame(betas)
df_pk = pd.DataFrame(pk)

betas_train, betas_test, pk_train, pk_test = train_test_split(df_betas, df_pk, 
    test_size=TEST_SIZE, random_state=RANDOM_STATE)

test_index = betas_test.index.values

##############################################################################
## SAVE PREPROCESSED DATA

# Create directory it doesn't exist
saving_path = 'data/' + SCENARIO + '/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)
    
# Save scaler
dump(scaler_output, open('data/' + SCENARIO + 
                         '/scaler_' + PC_STRATEGY + '.pkl', 'wb'))

np.save(saving_path + 'betas_' + PC_STRATEGY + '_train.npy', 
        betas_train.values)
np.save(saving_path + 'betas_' + PC_STRATEGY + '_test.npy',
        betas_test.values)

np.save(saving_path + 'pk_' + PC_STRATEGY + '_train.npy',
        pk_train.values)
np.save(saving_path + 'pk_' + PC_STRATEGY + '_test.npy',
        pk_test.values)

np.save(saving_path + 'SE_' + PC_STRATEGY + '_test.npy',
        SE[:,test_index])

np.save(saving_path + 'bk_' + PC_STRATEGY + '_test.npy',
        bk[:,test_index])
np.save(saving_path + 'ck_' + PC_STRATEGY + '_test.npy',
        ck[:,:,test_index])
np.save(saving_path + 'sigma2_' + PC_STRATEGY + '_test.npy',
        sigma2[:,test_index])