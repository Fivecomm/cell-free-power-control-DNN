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
import h5py
import matplotlib.pyplot as plt
from pickle import load

##############################################################################
## PARAMETERS TO SET UP

# Cell-free scenario
SCENARIO = 'scenario_01'

# Power control strategy: 'maxmin', 'sumSE', 'FPC'
PC_STRATEGY = 'maxmin'

# Simulations path
SIM_PATH = 'dataset/'

##############################################################################
## FUNCTIONS

def functionComputeSE(bk, ck, sigma2, prelog_factor, p):
    """
    Compute uplink SE from equation (7.1) in [1].

    Parameters:
    bk (np.array): Vector of length K with elements b_k in (7.2).
    ck (np.array): Matrix with dimension K x K where (i,k) is the c_{ki} in 
    (7.3)-(7.4).
    sigma2 (np.array): Vector of length K with elements \(\sigma_k^2\) in (7.5).
    preLogFactor (float): Pre-log factor in the SE expression of Theorem 5.2.
    p (np.array): Power coefficients.

    Returns:
    SE (np.array): Spectral efficiency.

    References:
    [1] Özlem Tuğfe Demir, Emil Björnson, and Luca Sanguinetti (2021) 
        “Foundations of User-Centric Cell-Free Massive MIMO”, 
        Foundations and Trends in Signal Processing: Vol. 14, No. 3-4,
        pp. 162-472. DOI: 10.1561/2000000109.
    """

    # Compute SINR
    SINR = np.divide(np.multiply(bk,p), np.matmul(ck.T, p) + sigma2)

    # Compute SE
    SE = prelog_factor * np.log2(1 + SINR)

    return SE

def cdf(data):
    """
    Compute the Cumulative Distribution Function (CDF) of the input data.

    Parameters:
    data (np.array): The input data as a numpy array.

    Returns:
    sorted_data (np.array): The input data sorted in ascending order.
    p (np.array): The values of the CDF for the corresponding elements in 
    sorted_data.
    """

    # Sort the data in ascending order
    sorted_data = np.sort(data, axis=0)

    # Calculate the proportional values of samples. This is done by creating an
    # array of indices (from 0 to len(data)-1), dividing each index by 
    # (len(data) - 1), and scaling by 1.0 to ensure the result is a floating 
    # point number. This gives us the percentile ranks of the data which we 
    # will use as the CDF values.
    p = 1. * np.arange(len(data)) / (len(data) - 1)

    # Return the sorted data and the corresponding CDF values
    return sorted_data, p

##############################################################################
## LOAD DATA

# Load prelog factor
with h5py.File(SIM_PATH + SCENARIO + '/setup.mat', 'r') as f:
        data = f.get('preLogFactor') 
        prelog_factor = np.array(data)
        data = f.get('L')   # number of APs
        L = np.array(data)  
        L = int(L.item())

# Load test data
pk_test = np.load('data/' + SCENARIO + '/pk_' + PC_STRATEGY + '_test.npy')
SE_test = np.load('data/' + SCENARIO + '/SE_' + PC_STRATEGY + '_test.npy')
bk_test = np.load('data/' + SCENARIO + '/bk_' + PC_STRATEGY + '_test.npy')
ck_test = np.load('data/' + SCENARIO + '/ck_' + PC_STRATEGY + '_test.npy')
sigma2_test = np.load('data/' + SCENARIO + '/sigma2_' + PC_STRATEGY + 
                      '_test.npy')

# Load DNN results
pk_dnn = np.load('results/' + SCENARIO + '/pk_' + PC_STRATEGY + '_output.npy')

##############################################################################
## COMPUTE SE

nbr_test_setups = pk_dnn.shape[0]
K = pk_dnn.shape[1] # number of UEs
SE_dnn = np.zeros((K,nbr_test_setups))

# Apply scaler to power coefficients
scaler = load(open('data/' + SCENARIO + 
                   '/scaler_' + PC_STRATEGY + '.pkl', 'rb'))
pk_test = scaler.inverse_transform(pk_test)
pk_dnn = scaler.inverse_transform(pk_dnn)

# Power coefficients from dB to mW
pk_test = np.transpose(1e3 * np.power(10, 0.1 * pk_test))
pk_dnn = np.transpose(1e3 * np.power(10, 0.1 * pk_dnn))

for setup in range(0, nbr_test_setups):
    SE_dnn[:,setup] = functionComputeSE(bk_test[:,setup], ck_test[:,:,setup], 
                                        sigma2_test[:,setup], prelog_factor, 
                                        pk_dnn[:,setup])

##############################################################################
## PLOT DISTRIBUTIONS 

# Plot histogram of the inputs
plt.hist(pk_test.flatten(), bins=100)
plt.gca().set(title='Histogram pk (test)', ylabel='Frequency')
plt.show()

# Plot histogram of the outputs
plt.hist(pk_dnn.flatten(), bins=100)
plt.gca().set(title='Histogram pk (dnn)', ylabel='Frequency')
plt.show()

##############################################################################
## PLOT CDF 

# Reshape 2D matrix SE into a vector
SE_test = np.reshape(SE_test, (-1, 1))
SE_dnn = np.reshape(SE_dnn, (-1, 1))    

# Calculate CDF
sorted_SE_test, cdf_test = cdf(SE_test)
sorted_SE_dnn, cdf_dnn = cdf(SE_dnn)

# Plot CDFs
plt.plot(sorted_SE_test, cdf_test, color='k', label=PC_STRATEGY)
plt.plot(sorted_SE_dnn, cdf_dnn, color='r', label='DNN')
plt.legend()
plt.xlabel('SE per UE (bit/s/Hz)')
plt.ylabel('CDF')
plt.title(SCENARIO + ' (UEs = ' + str(K) + ', APs = ' + str(L) + ')')
plt.grid(True)
plt.show()