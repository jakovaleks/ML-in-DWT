# RNN_BCV_DWT

This repository contains the code developed as part of the research project **"Investigating the Application of Recurrent Neural Networks and Blocked Cross-Validation for Modelling Conventional Drinking Water Treatment Processes"**.

A _synthetic_ data set representing one week of operation, formatted in the same way as the original data used for this project, can be found in `Data/stn_csv/stn_2401.csv`. You can test this code out using by running `example.py`, which will:
1. Merge all .csv files in `Data/stn_csv` into a single .npy file.
2. Remove anomalous data (based on the criteria described in `data_cleaning_mgmt.py`.
3. Create PyTorch tensors for model training and evaluation.
4. Quickly train an RNN on the training portion of the data.
5. Evaluate the RNN's performance on the production portion of the data.

## Description of the files/folders in this repository:

- `Main`
  - `_example.py` - To demonstrate quickly running the full process for going from raw data to trained and evaluated models, as described above.
  - `data_import_mgmt.py` - For importing data from their original .csv files into Python-compatible formats.
  - `data_cleaning_mgmt.py` - For removing anomalous data from the data set. See the comments in this file to see the criteria on which data were removed.
  - `data_torch_mgmt.py` - For converting NumPy arrays into PyTorch tensors that could be used for model training and evaluation.
  - `models.py` - Contains the MLP, LSTM, and GRU models; as described in the paper.
  - `train_rnn.py` - For training and testing the LSTM and GRU.
  - `train_mlp.py` - For training and testing the MLP.
  - `final_test.py` - For assessing the performance of trained models on the production data set.
  - `_production_summary.txt` - Created in final_test.py, contains the RMSE and MAE of each model as evaluated on the production set.
- `Data` - Contains the files holding the data for model training. The NumPy arrays created in `data_import_mgmt.py` and `data_cleaning_mgmt.py` are not stored in any subfolders.
  - `stn_csv` - Contains the raw .csv files which each contain one month of operating data. In this case there is only one file containing synthetic data.
  - `Torch` - Contains .pt files which are tensors for model training/evaluation.
- `Models` - Contains the trained models which each has:
  - A .pt file, which is the model itself that can be loaded using PyTorch.
  - A .txt file which summarizes the model parameters and its training results.
  - A .png file which is a graph of its training performance.
  - A _perf.npy file which contains the data used to plot the graph.
  - A _residuals.npy file which contains the MAE residuals for each point in each test block.
