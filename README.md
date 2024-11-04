# A Flexible Low-Complexity DNN Solution for Power Control in Cell-Free Massive MIMO

This repository contains the dataset and code to reproduce results of the following papers:

<<<<<<< HEAD
G. García-Barrios, M. Fuentes, and D. Martín-Sacristán, "A Flexible Low-Complexity DNN Solution for Power Control in Cell-Free Massive MIMO,"  *2024 IEEE 35th Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)*, Valencia, Spain, 2024.

G. García-Barrios, M. Fuentes and D, Martín-Sacristán, "Data-Driven Energy Efficiency Modeling in Large-Scale Networks: An Expert Knowledge and ML-Based Approach," in *IEEE Transactions on Machine Learning in Communications and Networking*. [Submitted]
=======
Guillermo García-Barrios, Manuel Fuentes, David Martín-Sacristán , "A Flexible Low-Complexity DNN Solution for Power Control in Cell-Free Massive MIMO,"  *2024 IEEE 35th Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)*, Valencia, Spain, 2024. [Presented]
>>>>>>> 57c89f6604b98c4419f9ab54b7dafc55a679afec

## Abstract of the Paper

Cell-free massive multiple-input multiple-output (MIMO) is a technology that combines ultra-dense cellular networks and massive MIMO to enhance wireless transmission efficiency, being a promising solution for future communication scenarios. However, its implementation poses a significant computational load, which can hinder real-time implementation. The use of machine learning comes as a great solution to reduce this computational complexity. In this context, the present work explores the application of a supervised, low-complexity deep neural network (DNN) for power control, aiming to significantly reduce computational costs. The DNN was evaluated across various scenarios, generating a comprehensive dataset. The proposed DNN outperforms existing methods, demonstrating superior performance with three power optimization schemes: max-min spectral efficiency (SE) fairness, sum SE maximization, and fractional power control (FPC). Importantly, these results are consistent across diverse propagation conditions. A noteworthy outcome of this study is the achievement of a three-orders-of-magnitude reduction in computation time when compared to conventional techniques, paving the way for the practical implementation of cell-free massive MIMO systems.

## Content of Code Package

This code package is structured as follows:

- `data_preprocessing.py`: This script performs the data preprocessing before training the DNN model. Preprocessed data is saved in `data/` folder.
- `train.py`: This script trains the DNN and saves the model in the `models/` folder.
- `test.py`: This script tests the DNN. The output power coefficients are saved in the `results/` folder.
- `test_generalization.py`: This script tests the DNN for a different scenario from which has been trained. An special preprocessing is applied to ensure compatibility. The output power coefficients are saved in the `results/` folder.
- `plot_results.py`: This script plots the results of the testing data.
- `plot_results_generalization.py`: This script plots the results of the testing data for the generalization part.

See each file for further documentation.

**NOTE:** The directories mentioned earlier should be created manually.

# Associated dataset

This repository is associated with a dataset that contains the simulation of 12 distinct cell-free massive MIMO scenarios, where each scenario includes 20,000 setups. The simulations take into account three power control optimization schemes: max-min Spectral Efficiency (SE) fairness, sum SE maximization, and Fractional Power Control (FPC).

The dataset is available at [https://zenodo.org/records/10691343](https://zenodo.org/records/10691343).

**NOTE:** The downloaded files should be placed in a directory named `dataset/`.

# Acknowledgments

This work is supported by the grant from the Spanish ministry of economic affairs and digital transformation and of the European Union – NextGenerationEU [UNICO-5G I+D/AROMA3D-Earth] (TSI-063000-2021-69).

# License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our conference paper listed above.
