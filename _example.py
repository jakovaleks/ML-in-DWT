# This file shows an example of how to use the included files.
# It covers importing the data, cleaning it, and converting it to PyTorch tensors.
# It also trains and tests an RNN model on the data.

# Consult the individual files for full comments.

# This is all done starting with only the file "stn_2401.csv",
# which is a synthetic data set containing one week of operating data.

import fnmatch, os, torch

# Import the other files
import data_import_mgmt
import data_cleaning_mgmt
import data_torch_mgmt
import train_rnn
import final_test

# File location/name definitions
f_data = "Data/"
f_csv = f"{f_data}stn_csv/"
f_torch = f"{f_data}Torch/"
f_model = "Models/"
f_output = "Final Test/"

# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# -----------------

#%% Step 1: Import the data from the .csv file and convert it into a .npy file,
# and convert the dates into decimal dates (data_import_mgmt.py)

# Load a list of all the monthly .csv files containing data
list_csv = sorted(fnmatch.filter(os.listdir(f_csv), "stn_*.csv"))
start_date = list_csv[0][list_csv[0].find('_')+1:list_csv[0].find('.')]
end_date = list_csv[-1][list_csv[-1].find('_')+1:list_csv[-1].find('.')]
file_npy = f"{f_data}stn_{start_date}-{end_date}.npy"

# Append "_decimal" for the modified file with decimal timestamps
file_npy_dec = file_npy[:file_npy.rfind('.')] + '_decimal' + file_npy[file_npy.rfind('.'):]

# Main
data_import_mgmt.func_csv_to_npy(f_csv, list_csv, file_npy)
data_import_mgmt.func_npy_to_npy_decimal(file_npy, file_npy_dec)

# -----------------

#%% Step 2: Clean the data to remove sensor errors and other anomalies.
# These changes are fully documented in the file (data_cleaning_mgmt.py)

# Load the _decimal file
file_npy_dec = f"stn_{start_date}-{end_date}_decimal.npy"

# Append "_mod" for modified file after cleaning
file_npy_mod = file_npy_dec[:file_npy_dec.rfind('.')] + '_mod' + file_npy_dec[file_npy_dec.rfind('.'):]

# Main
data_cleaning_mgmt.func_clean_data(f_data + file_npy_dec, f_data + file_npy_mod)

# -----------------

#%% Step 3: Convert the 2D .npy arrays into 3D .pt tensors of size
# [num_examples, num_time_steps, num_variables] (data_torch_mgmt.py)

# Main
total_steps = 45
discontinuous_steps = 5
filename = data_torch_mgmt.func_make_rnn_tensors(file_npy_mod, start_date, end_date, total_steps, discontinuous_steps)
finalnames = data_torch_mgmt.func_consolidate_tensors(f_torch, filename, start_date, end_date, split_date='2024-01-07 00:00')
# split_date must be in the form YYYY-MM-DD HH-MM
finalnames_n, master_vals = data_torch_mgmt.func_normalize_variables(f_torch, finalnames)

# -----------------

#%% Step 4: Train an RNN model (train_rnn.py)

# Main

batch_size = 64
output_steps = 15
# You can put multiple combinations of rnn type, number of layers,
# and hidden layer size and automatically iterate through all of them
for algo, layer, hidden in [("GRU", 2, 32)]:
        trainer = train_rnn.rnn_trainer(algo, f_torch, f_model)
        trainer.initialize(
                           training_data_filename=finalnames_n[0],
                           master_values_filename=master_vals,
                           batch_size=batch_size,
                           output_steps=output_steps,
                           num_training_blocks=4,    # Represents N-2 training blocks for BCV
                           # If blocked_crossval=False, the fraction (N-2)/N is the training set
                           hidden_dims=hidden,
                           num_layers=layer,
                           teacher_forcing_ratio=0.5,
                           )
        trainer.train(
                      learning_rate=5e-4,
                      max_iters_per_block=25,
                      max_iters_increasing_validation_loss=5,
                      weight_decay=1e-8,
                      gradient_clipping_norm=1,
                      blocked_crossval=True    # Enables BCV (set to False for holdout)
                      )
        trainer.save()
        
# -----------------

#%% Step 5: Evaluate the model's performance on the production set (final_test.py)

tester = final_test.model_tester(f_torch, f_model)
tester.initialize(
                  production_data_filename=finalnames_n[1],
                  master_values_filename=master_vals,
                  batch_size=batch_size,
                  output_steps=output_steps
                  )
tester.test()