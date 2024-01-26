# This file is for preparing the PyTorch tensors of the cleaned up data for
# training and also includes utilities for loading and batching the tensors.

import torch
import fnmatch
import numpy as np
from torchdata.datapipes.iter import IterableWrapper
from data_import_mgmt import func_year_frac

# File locations
f_root = "/home/user1/Documents/ML_DWT/"
f_data = "Data/"
f_torch = "Torch/"
fileloc = f"{f_root}{f_data}"
fileloc_torch = f"{f_root}{f_data}{f_torch}"
filename_npy_mod = "stn_1707_2111_decimal_mod.npy"
start_date, end_date = '1707', '2111'
    
# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# FINAL INDICES
# 0. Settled turbidity
# 1. Influent turbidity
# 2. Influent conductivity
# 3. Coagulation basin average pH
# 4. Influent temperature
# 5. Alum dose
# 6. Flow rate in settling tank

# ================================================================ #
# stn_1707_2111 indices ========================================== #
# 0. decimal_time; 1. temperature_in; 2. turbidity_in; =========== #
# 3. conductivity; 4. ph_in; 5. scm; 6. coagulation_ph_1; ======== #
# 7. coagulation_ph_2; 8. actiflo_1_turbidty; 9. actiflo_1_tss; == #
# 10. actiflo_2_turbidity; 11. actiflo_2_tss; ==================== #
# 12. actiflo_3_turbidity; 13. actiflo_3_tss; ==================== #
# 14. alum; 15. pax; 16. actiflo_1_polymer; ====================== #
# 17. actiflo_2_polymer; 18. actiflo_3_polymer; 19. silica; ====== #
# 20. inflow, 21. actiflo_1_flow; 22. actiflo_2_flow; ============ #
# 23. actiflo_3_flow; ============================================ #
# ================================================================ #


# Function to create 3D tensors of size (examples, steps, dims) for training RNNs
def func_make_rnn_tensors(total_steps=45, discontinuous_steps=5, additional_vars=[]):
    
    # Load data and create a PyTorch tensor
    data_np = np.load(fileloc + filename_npy_mod)
    data_len, total_dims = data_np.shape    
    data_pt = torch.tensor(data_np, dtype=torch.float64, device=device)
    
    # Create the tensors
    print('Creating RNN tensors...')
    # Create a new tensor of size (examples, steps, dims)
    tensor_pt = torch.zeros((data_len - total_steps, total_steps, total_dims), dtype=torch.float64, device='cpu')
    # Iterate through (examples) and put (steps) consecutive data points into each
    print_every = int((data_len - total_steps) / 20)
    for i in range(data_len - total_steps):
        tensor_pt[i] = data_pt[None, i:i + total_steps, :]
        if (i % print_every == 0) & (i > 0):
            print(f'{int(np.around((i + 1) / (data_len - total_steps), 2) * 100)}% complete.')
    print('Done.\n')
    
    # Delete examples where all total_steps steps are not continuous,
    # and where the first or last element is within discontinuous_steps
    # steps of a discontinuity (since the data seem to take a few minutes to recover)
    print('Removing discontinuous tensors...')
    del_idx = torch.zeros(data_len - total_steps, dtype=torch.bool)
    
    # Size of one time step (one minute) for a standard/leap year respectively
    single_step, single_step_leap = 1/365/24/60, 1/366/24/60
    
    for i in range(data_len - total_steps):
        # Calculate the expected difference in decimal time based on if it is a leap year or not
        expected_time = (total_steps - torch.sum(func_leap_year(tensor_pt[i, :, 0]))) * single_step + \
                        torch.sum(func_leap_year(tensor_pt[i, :, 0])) * single_step_leap
        
        # Flag an example for deletion if its last time step is higher
        # than the expected value (i.e. more than total_steps in the future)
        # or if it is within discontinuous_steps of such an example
        if (tensor_pt[i, -1, 0] - tensor_pt[i, 0, 0]) > expected_time:
            del_idx[np.max((i - discontinuous_steps, 0)) : \
                    np.min((i + discontinuous_steps, len(del_idx)))] = 1
                
        if (i % print_every == 0) & (i > 0):
            print(f'{int(np.around((i + 1) / (data_len - total_steps), 2) * 100)}% complete.')
            
    # Delete the data that was flagged as being discontinuous
    del_idx = ~del_idx
    tensor_pt = tensor_pt[del_idx]
    print('Done.\n')
    
    # Save tensor
    filename = f"TEMP_{start_date}-{end_date}_{total_steps}x{total_dims}.pt"
    torch.save(tensor_pt, fileloc_torch + filename)
    print(f'Tensor saved to {fileloc_torch}{filename}.\n')
    
    return filename


# Function to turn tensor outputs into (out_steps x 1) [as opposed to (out_steps x 3]
# and remove offline clarifiers
def func_consolidate_tensors(fileloc_torch, filename, split_date=None):
    tensor_pt = torch.load(fileloc_torch + filename).to(device='cpu')

    # Goals here:
    # 1) Remove influent pH, and average the two coagulation basin pH readings
    # 2) Have turbidity and flow rate for each clarifier as its own example,
    #    along with the other common variables in each example
    # 3) Remove inflow, pax, TSS, the date column, and polymer
    # 4) Remove offline clarifiers
    # 5) For simplicity, ensure that turbidity_out is at the first index
    # 6) Split the tensor into training/production files (if desired,
    #    in this case we use it for comparing BCV and holdout)
    
    # Get dates for filename(s)
    start_date = tensor_pt[0, 0]
    
    print('Changing tensors from three outputs to one...')
    data_len, total_steps, total_dims = tensor_pt.shape
    
    # Check split_date (if provided) to ensure it is in YYYY-MM-DD HH:MM form
    if split_date is not None:
        temp_list = []
        temp_list.append(split_date)
        temp_filter = fnmatch.filter(temp_list, '????-??-?? ??:??')
        assert temp_filter, 'split_date must be in the form YYYY-MM-DD HH:MM'
        mid_date = split_date[2:4] + split_date[5:7]
    
    # Average coagulation basin readings and save to column 6
    tensor_pt[:, :, 6] = torch.mean(tensor_pt[:, :, [6, 7]], axis=2).to('cpu')
    
    # Indicate the desired columns to keep
    # 0. date (kept for now for sorting, removed at the end);
    # 1. temp; 2. turb; 3. cond; 6. ph_coag; 14. alum;
    common_dims = [0, 1, 2, 3, 6, 14]
    # 8/10/12. act1/2/3_turbidity; 21/22/23. act1/2/3_flow
    t1_dims = [*common_dims, *[8, 21]]
    t2_dims = [*common_dims, *[10, 22]]
    t3_dims = [*common_dims, *[12, 23]]
    t1_dims.sort(), t2_dims.sort(), t3_dims.sort()
    
    # Create a new tensor for each clarifier and delete instances where
    # the clarifier is offline during the latter half of the window
    t1 = tensor_pt[:, :, t1_dims].detach().clone()
    t1 = t1[(t1[:, int(total_steps/2):, -1].all(1) > 0), :, :]
    t2 = tensor_pt[:, :, t2_dims].detach().clone()
    t2 = t2[(t2[:, int(total_steps/2):, -1].all(1) > 0), :, :]
    t3 = tensor_pt[:, :, t3_dims].detach().clone()
    t3 = t3[(t3[:, int(total_steps/2):, -1].all(1) > 0), :, :]
    
    # The data for Actiflo 3 behaves erratically for Sept 2021 - Nov 2021
    # so it is removed here
    t3_cutoff = func_year_frac('2021-09-01 00:00')
    t3 = t3[(t3[:, -1, 0] < t3_cutoff), :, :]
    
    # Concatenate the tensors for the 3 clarifiers and sort by date
    t = torch.cat((t1, t2, t3), dim=0).to('cpu')
    t_idx = torch.argsort(t[:, 0, 0]).to('cpu')
    t = t[t_idx, :, :]
    
    # Split into training/production sets if split_date was set
    if split_date is not None:
        t_train = t[(t[:, -1, 0] < split_date), :, :]
        t_prod = t[(t[:, -1, 0] >= split_date), :, :]
        # Remove the date column and move effluent turbidity (index 4) to index 0
        t_train = t_train[:, :, 1:]
        t_train[:, :, [0, 4]] = t_train[:, :, [4, 0]]
        t_prod = t_prod[:, :, 1:]
        t_prod[:, :, [0, 4]] = t_prod[:, :, [4, 0]]
    else:
        # Remove the date column and move effluent turbidity (index 4) to index 0
        t = t[:, :, 1:]
        t[:, :, [0, 4]] = t[:, :, [4, 0]]
    
    # Save
    finalname = []
    if split_date is not None:
        _, total_steps, final_dims = t_train.shape
        fname1 = f"TRAIN_{start_date}-{mid_date}_{total_steps}x{final_dims}.pt"
        fname2 = f"PROD_{mid_date}-{end_date}_{total_steps}x{final_dims}.pt"
        torch.save()
        print(f'Tensors saved to {fileloc_torch}{fname1} and {fileloc_torch}{fname2}.\n')
        finalname.append(fname1)
        finalname.append(fname2)
    else:
        _, total_steps, final_dims = t.shape
        fname1 = f"MOD_{start_date}-{end_date}_{total_steps}x{final_dims}.pt"
        torch.save(t, fileloc_torch + finalname)
        print(f'Tensor saved to {fileloc_torch}{finalname}.\n')
        finalname.append(fname1)
    return finalname
  

# Function for 0-1 data normalization
def func_normalize_variables(fileloc_torch, filename):
    for file in filename:
        print('Normalizing variables...')
        # Load original file
        t_orig = torch.load(fileloc_torch + file)
        # Create empty tensor for normalized variables
        t = torch.empty(t_orig.shape)
        n_dims = t_orig.shape[-1]
        
        # For each variable, find min and max value, then transform
        # all values of that variable into the range 0-1
        for i in range(n_dims):
            minval = torch.min(t_orig[:, :, i])
            maxval = torch.max(t_orig[:, :, i])
            t[:, :, i] = (t_orig[:, :, i] - minval)/(maxval - minval)
        
        # Save as new file with "_n"(ormalized) appended to the filename
        file = filename.replace('.pt', '_n.pt')
        torch.save(t, fileloc_torch + file)
        print(f'Tensor saved to {fileloc_torch}{filename}.\n')
    
    
# Function to return normalized data to original
def func_unnormalize_variables(data):
    # Create empty tensor
    unnorm = torch.zeros(data.size())
    data_len, total_steps, n_dims = data.shape
    # Load original data
    t_orig = torch.load(fileloc_torch + 'MASTER_{start_date}-{end_date}_{total_steps}x{n_dims}.pt')
    
    # Iterate through all variables
    for i in range(n_dims):
        # Find min and max values in original data
        minval = torch.min(t_orig[:, :, i])
        maxval = torch.max(t_orig[:, :, i])
        # Transform back to original values (input data may be a 2D or 3D array)
        if len(unnorm.shape) == 2:
            unnorm[:, i] = (data[:, i] * (maxval - minval)) + minval
        if len(unnorm.shape) == 3:
            unnorm[:, :, i] = (data[:, :, i] * (maxval - minval)) + minval          

    return unnorm
    

# Function to load tensors as a 4D array, where the first dimension of the array
# represents one of n blocks for time series validation
def func_loader(fileloc_torch, finalname, num_blocks=14):
    tensor_3d = torch.load(fileloc_torch + finalname).to('cpu')
    num_pts, total_steps, total_dims = tensor_3d.shape
    last_pt = num_pts - (num_pts % num_blocks) # <num_blocks pts are lost
    pts_per_block = int(last_pt / num_blocks)
    tensor_4d = tensor_3d[:last_pt, :, :].view(num_blocks, pts_per_block, total_steps, total_dims)
    return tensor_4d


# Function to split a tensor into minibatches
def func_batcher(batch_size, tensor_size):
    batcher = IterableWrapper(range(tensor_size))
    batcher = batcher.batch(batch_size=batch_size, drop_last=True)
    # Returns an iterable of minibatches for one epoch
    return iter(batcher)


# Function to check if a date is in a leap year or not
def func_leap_year(date):
    # Returns TRUE if the given year is a leap year, FALSE if it is a standard year
    return (np.floor(date) % 4 == 0) & (np.floor(date) % 100 > 0) | (np.floor(date) % 400 == 0)


#%% MAIN
if __name__ == '__main__':
    total_steps, discontinuous_steps, additional_vars = 45, 5, []
    filename = func_make_rnn_tensors(total_steps, discontinuous_steps, additional_vars)
    finalname = func_consolidate_tensors(fileloc_torch, filename, func_split_date='2020-09-01 00:00')
    # split_date must be in the form YYYY-MM-DD HH-MM
    func_normalize_variables(fileloc_torch, finalname)