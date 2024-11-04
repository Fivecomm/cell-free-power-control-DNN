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

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pickle import load
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson

##############################################################################
## PARAMETERS TO SET UP

# Cell-free scenario
SCN_TRAIN = 'scenario_07'
SCN_TEST = 'scenario_05'

# Power control strategy: 'maxmin', 'sumSE', 'FPC'
PC_STRATEGY = 'FPC'

# Simulations path
SIM_PATH = '../../MATLAB/cell-free_simulated_scenarios/'

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

# Load test data info 
with h5py.File('data/' + SCN_TEST + '.mat', 'r') as f:
    data = f.get('preLogFactor') 
    prelog_factor_test = np.array(data)
    data = f.get('L')   # number of APs
    L = np.array(data)  
    L_test = int(L.item())
    data = f.get('K')   # number of UEs
    K = np.array(data)  
    K_test = int(K.item())

# Load train data info   
with h5py.File('data/' + SCN_TRAIN + '.mat', 'r') as f:
    data = f.get('L')   # number of APs
    L = np.array(data)  
    L_train = int(L.item())
    data = f.get('K')   # number of UEs
    K = np.array(data)  
    K_train = int(K.item())

# Load test data
pk_test = np.load('data/' + SCN_TEST + '/pk_' + PC_STRATEGY + '_test.npy')
SE_test = np.load('data/' + SCN_TEST + '/SE_' + PC_STRATEGY + '_test.npy')
bk_test = np.load('data/' + SCN_TEST + '/bk_' + PC_STRATEGY + '_test.npy')
ck_test = np.load('data/' + SCN_TEST + '/ck_' + PC_STRATEGY + '_test.npy')
sigma2_test = np.load('data/' + SCN_TEST + '/sigma2_' + PC_STRATEGY + 
                      '_test.npy')

# Load DNN results
pk_dnn_test = np.load('results/train_' + SCN_TRAIN + '-test_' + SCN_TEST + 
                      '/pk_' + PC_STRATEGY + '_output.npy')
pk_dnn = np.load('results/' + SCN_TEST + '/pk_' + PC_STRATEGY + '_output.npy')

##############################################################################
## COMPUTE SE

nbr_test_setups = pk_dnn_test.shape[0]
SE_dnn_test = np.zeros((K_test,nbr_test_setups))
SE_dnn = np.zeros((K_test,nbr_test_setups))

# Apply scaler to power coefficients
scaler = load(open('data/' + SCN_TEST + '/scaler_' + PC_STRATEGY + '.pkl', 'rb'))
pk_test = scaler.inverse_transform(pk_test)
pk_dnn = scaler.inverse_transform(pk_dnn)

scaler_train = load(open('data/' + SCN_TRAIN +
                         '/scaler_' + PC_STRATEGY + '.pkl', 'rb'))
pk_dnn_test= scaler_train.inverse_transform(pk_dnn_test)

# Power coefficients from dB to mW
pk_test = np.transpose(1e3 * np.power(10, 0.1 * pk_test))
pk_dnn_test = np.transpose(1e3 * np.power(10, 0.1 * pk_dnn_test))
pk_dnn = np.transpose(1e3 * np.power(10, 0.1 * pk_dnn))

# Delete values
pk_dnn_test = np.delete(pk_dnn_test, slice(K_test,K_train), axis=0)

for setup in range(0, nbr_test_setups):
    SE_dnn_test[:,setup] = functionComputeSE(bk_test[:,setup], 
                                             ck_test[:,:,setup],
                                             sigma2_test[:,setup],
                                             prelog_factor_test,
                                             pk_dnn_test[:,setup])
    
    SE_dnn[:,setup] = functionComputeSE(bk_test[:,setup], 
                                        ck_test[:,:,setup],
                                        sigma2_test[:,setup],
                                        prelog_factor_test,
                                        pk_dnn[:,setup])

##############################################################################
## PLOT DISTRIBUTIONS 

# Plot histogram of the inputs
plt.hist(pk_test.flatten(), bins=100)
plt.gca().set(title='Histogram pk (test)', ylabel='Frequency')
plt.show()

# Plot histogram of the outputs (test)
plt.hist(pk_dnn_test.flatten(), bins=100)
plt.gca().set(title='Histogram pk (dnn test)', ylabel='Frequency')
plt.show()

# Plot histogram of the outputs (train)
plt.hist(pk_dnn.flatten(), bins=100)
plt.gca().set(title='Histogram pk (dnn)', ylabel='Frequency')
plt.show()

##############################################################################
## PLOT CDF 

# Reshape 2D matrix SE into a vector
SE_test = np.reshape(SE_test, (-1, 1))
SE_dnn_test = np.reshape(SE_dnn_test, (-1, 1))
SE_dnn = np.reshape(SE_dnn, (-1, 1))  

# Calculate CDF
sorted_SE_test, cdf_test = cdf(SE_test)
sorted_SE_dnn_test, cdf_dnn_test = cdf(SE_dnn_test)
sorted_SE_dnn, cdf_dnn = cdf(SE_dnn)

kolmogorov = ks_2samp(sorted_SE_dnn.flatten(), sorted_SE_dnn_test.flatten())
print('KS Test - p-value:', kolmogorov.pvalue, ' - D: ', kolmogorov.statistic)
cramer = cramervonmises_2samp(sorted_SE_dnn.flatten(), sorted_SE_dnn_test.flatten())
print('Cramer Test - p-value:', cramer.pvalue, ' - D: ', cramer.statistic)

# Plot CDFs
plt.plot(sorted_SE_test, cdf_test, color='k', label=PC_STRATEGY)
plt.plot(sorted_SE_dnn_test, cdf_dnn_test, color='r', label='DNN (test)')
plt.plot(sorted_SE_dnn, cdf_dnn, linestyle='dotted', color='b', label='DNN')
plt.legend()
plt.xlabel('SE per UE (bit/s/Hz)')
plt.ylabel('CDF')
plt.title('Train: ' + SCN_TRAIN + ' (UEs = ' + str(K_train) + ', APs = ' + 
          str(L_train) + ')\n'
          'Test: ' + SCN_TEST + ' (UEs = ' + str(K_test) + ', APs = ' + 
          str(L_test) + ')')
plt.grid(True)
plt.savefig('results/train_' + SCN_TRAIN + '-test_' + SCN_TEST + '/cdf_' +
            PC_STRATEGY + '.png')
plt.show()