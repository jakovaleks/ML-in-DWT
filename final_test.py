# This file is for testing the final models.

import torch
import os
import natsort
import numpy as np
from data_torch_mgmt import func_batcher
from torch import nn

# File locations
f_root = "/home/user1/Documents/ML_DWT/"
f_data = "Data/Torch/"
f_model = "Models/"
f_output = "Final Test/"
fileloc_data = f"{f_root}{f_data}"
fileloc_model = f"{f_root}{f_model}"
fileloc_output = f"{f_root}{f_output}"
    
# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# Define tensor and batch sizes
total_dims = 7
input_steps = 30
output_steps = 15
batch_size = 64

# Load production data set
prod_data = torch.load(fileloc_data + 'PROD_2009-2111_45x7_n.pt')
data_in = prod_data[:, :input_steps, :].to(device)
true_out = prod_data[:, input_steps:, 0].to(device)
examples = len(prod_data)

# Load names of all the models to be tested
model_tags = ['MLP_BCV', 'LSTM_BCV', 'GRU_BCV', 'MLP_HO', 'LSTM_HO', 'GRU_HO']
# We have six different model types (the above tags, MLP/LSTM/GRU trained with BCV/HO)
# and 9 sizes of each model
model_names = np.empty((6, 9), dtype=object)
for i, tag in enumerate(model_tags):
    j = 0
    for name in os.listdir(fileloc_model + tag):
        if name.endswith('.pt'):
            model_names[i, j] = name
            j += 1
    model_names[i, :] = natsort.natsorted(model_names[i, :])

# Function to calculate loss in NTU
def func_loss(self, preds, actual, testing=False):
    loss = nn.MSELoss()
    loss_MAE = nn.L1Loss(reduction='none')
    # to avoid having to load the original data file every time to find
    # the maximum turbidity in the data set, simply find it once and 
    # paste it into here for future reuse
    turb_max = 10.0200002193451 
    # Since the minimum turbidity is 0, the un-normalization calculation
    # is simplified to simply multiplying the values by the maximum
    # turbidity in the original data set
    rmse = torch.sqrt(loss(turb_max * preds, turb_max * actual)).to(device)
    mae = loss_MAE(turb_max * preds, turb_max * actual)
    return rmse, mae

#%% Production set test
# Produces a .txt file with the RMSE and MAE of each model on the production set
for i, tag in enumerate(model_tags):
    for model in model_names[i, :]:
        mdl = torch.load(fileloc_model + tag + '/' + model).to(device)
        loss_tst, residuals_tst = [], []
        batch_gen = func_batcher(batch_size, examples)
        
        with torch.no_grad():
            # Iterate through all the batches in this epoch
            for bat in batch_gen:
                if 'MLP' in model:
                    x_bat, y_bat = torch.reshape(data_in[bat, :, :], (batch_size, input_steps * total_dims)), true_out[bat, :]
                    preds_tst = (mdl.forward(x_bat)).to(device)
                    rmse_tst, mae_tst = func_loss(preds_tst, y_bat)
                else:
                    x_bat, y_bat = data_in[bat, :, :], true_out[bat, :]
                    mdl.teacher_forcing_ratio = 0
                    preds_tst = (mdl.forward(x_bat, y_bat)).to(device)
                    rmse_tst, mae_tst = func_loss(preds_tst, y_bat.view((batch_size, output_steps, 1)))
                loss_tst.append(rmse_tst.tolist())
                residuals_tst.append(mae_tst.tolist())
                
        final_rmse = np.mean(loss_tst)
        final_mae = np.mean(residuals_tst)
        
        with open(f"{fileloc_output}RMSE.txt", "a") as f:
            f.write(f"{tag:<11}{model:<53}{final_rmse:.4f}   {final_mae:.4f}\n")
        np.save(f"{fileloc_output}{tag}_{model}_residuals", np.array(residuals_tst).reshape((-1, output_steps)))
        
        print(f'Done {tag}_{model}.')
            
