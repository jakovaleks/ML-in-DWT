# RNN_BCV_DWT

This repository contains the code developed as part of the research project "Investigating the Application of Recurrent Neural Networks and Blocked Cross-Validation for Modelling Conventional Drinking Water Treatment Processes".

**The files in this repository are as follows:**

data_import_mgmt.py - For importing data from their original .csv files into Python-compatible formats.

data_cleaning_mgmt.py - For removing anomalous data from the data set. See the comments in this file to see the criteria on which data were removed.

data_torch_mgmt.py - For converting NumPy arrays into PyTorch tensors that could be used for model training.

models.py - Contains the MLP, LSTM, and GRU models; as described in the paper.

train_rnn.py - For training and testing the LSTM and GRU.

train_mlp.py - For training and testing the MLP.

final_test.py - For assessing the performance of trained models on the production data set.

results.py - For generating figures to be included in the paper.
