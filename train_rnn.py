# This file is for training RNN models.

import torch
import models
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from data_torch_mgmt import func_loader, func_batcher, func_unnormalize_variables

# File locations
f_data = "Data/Torch/"
f_model = "Models/"
    
# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# Class to run the training procedure
class rnn_trainer:
    def __init__(self, rnn_type, f_data, f_model):
        
        # Check rnn_type is allowed and load data
        rnn_list = ['LSTM', 'GRU']
        assert rnn_type in rnn_list, f'rnn_type must one of: {rnn_list}'
        self.rnn_type = rnn_type
        
        # Specify folders to load/save data/models to/from
        self.f_data = f_data
        self.f_model = f_model
    
    # Load the training data and initialize the model
    def initialize(self,
                   training_data_filename,
                   master_values_filename,
                   batch_size=64,
                   output_steps=15,
                   num_training_blocks=12,
                   hidden_dims=64,
                   num_layers=1,
                   teacher_forcing_ratio=0.5,
                   ):
        
        self.training_data_filename = training_data_filename
        self.master_values_filename = master_values_filename
        self.batch_size = batch_size
        self.output_steps = output_steps
        self.num_training_blocks = num_training_blocks
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Load the data as a 4D array of size (blocks, examples_per_block, steps, variables)
        self.xy = func_loader(self.f_data, self.training_data_filename, self.num_training_blocks + 2)
        _, self.points_per_block, self.total_steps, self.total_dims = self.xy.shape
        self.input_steps = self.total_steps - self.output_steps
        
        # Set model parameters
        self.mdl_parameters = (self.total_dims,
                               self.hidden_dims,
                               self.num_layers,
                               self.batch_size,
                               self.input_steps,
                               self.output_steps,
                               self.teacher_forcing_ratio,
                               )
        self.mdl_parameter_names = ['Total dimensions',
                                    'Hidden dimensions',
                                    'Hidden layers',
                                    'Batch size',
                                    'Input time steps',
                                    'Output time steps',
                                    'Teacher forcing ratio',
                                    ]
        
        # Load a model based on the above parameters
        self.mdl = models.set_model_type(self.rnn_type, self.mdl_parameters).to(device)
        
        # Load master (unnormalized) values for model evaluation
        self.master_vals = torch.load(self.f_data + self.master_values_filename)
        
    # Train the model
    def train(self,
              learning_rate=5e-4,
              max_iters_per_block=250,
              max_iters_increasing_validation_loss=10,
              weight_decay=1e-8,
              gradient_clipping_norm=1,
              blocked_crossval=True
              ):
        
        # Load optimizer
        self.opt = optim.Adam(self.mdl.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.max_iters_per_block = max_iters_per_block
        self.blocked_crossval = blocked_crossval
        print(f'Training {self.mdl.desc}\nOptimization to be performed using {type(self.opt)}.\n')
        
        # Initialize loss logs for trn and val as empty lists
        self.loss_trn, self.loss_val = [], []
        
        # Initialize loss log for tst as a 2 column array [final_block_iter, test_set_rmse]
        self.loss_tst = np.zeros((self.num_training_blocks, 2))
        self.total_iters = 0
        self.residuals_tst = []
        
        # Train using blocked cross-validation
        start_time = time.perf_counter()
        self.stuck_in_zero_grad = False
        
        # If blocked_crossval == True:
        # Loop through n-2 blocks (on the last iteration, the valdiation
        # and test blocks are at the end of the data set)
        # If blocked_crossval == False:
        # Immediately assign the first n-2 blocks as the training set (i.e. equivalent
        # to a standard holdout cross-validation)
        # To achieve a 50/25/25 train/valid/test split, num_training_blocks should
        # be set to 2 since this will result in 4 total blocks
        if self.blocked_crossval == True:
            training_range = range(self.num_training_blocks)
        else:
            training_range = range(self.num_training_blocks - 1, self.num_training_blocks)
            
        for block in training_range:
            # Load training set for the current block
            xy_trn = self.xy[:block + 1, :, :, :].view(((block + 1) * self.points_per_block,
                                                        self.total_steps,
                                                        self.total_dims)
                                                        )
            x_trn = (xy_trn[:, :-self.output_steps, :]).to(device)
            y_trn = (xy_trn[:, -self.output_steps:, :]).to(device)
            
            # Load validation data for the current block
            xy_val = self.xy[block + 1, :, :, :].view((self.points_per_block,
                                                        self.total_steps,
                                                        self.total_dims)
                                                        )
            # Remove the first total_steps examples to avoid data leakage
            x_val = (xy_val[self.total_steps:, :-self.output_steps, :]).to(device)
            # y_val only needs the turbidity as teacher forcing is not used
            y_val = (xy_val[self.total_steps:, -self.output_steps:, 0]).to(device)
            
            # Shuffle the order of examples in x_trn and y_trn
            trn_idx = torch.randperm((block + 1) * self.points_per_block)
            x_trn = x_trn[trn_idx]
            y_trn = y_trn[trn_idx]
                
            # Iterate training loop until either max_iters_per_block is reached,
            # or validation loss is above the lowest loss for that block for
            # max_iters_increasing_validation_loss consectutive iterations
            for i in range(max_iters_per_block):
                self.mdl.train()
                
                # Set the teacher forcing ratio for training
                self.mdl.teacher_forcing_ratio = self.teacher_forcing_ratio
                
                # Create a "batch generator" that will load all the batches for
                # this iteration (epoch)
                batch_gen = func_batcher(self.batch_size,
                                         (block + 1) * self.points_per_block
                                         )
                
                # Empty the iteration loss list for this training iteration
                loss_iter = []
                
                # Iterate through all the training batches in this epoch
                for bat in batch_gen:
                    self.opt.zero_grad()
                    
                    # Load the input/output data in the batch
                    x_bat, y_bat = x_trn[bat, :, :], y_trn[bat, :, :]
                    
                    # Perform forward pass on the data
                    preds_trn = (self.mdl.forward(x_bat, y_bat)).to(device)
                    
                    # Calculate training loss on this batch
                    # We transform it back to NTU for meaningful interpretation!
                    rmse_trn, _ = self.func_loss(preds_trn, y_bat[:, :, 0].view((self.batch_size, self.output_steps, 1)), self.master_vals[0, :])
                                        
                    # Save the training loss to the list
                    loss_iter.append(rmse_trn.tolist())
                    
                    # Perform backwards pass
                    rmse_trn.backward()
                    torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), gradient_clipping_norm)
                    self.opt.step()
                    
                # Save the average loss for this iteration to the training loss list
                self.loss_trn.append(np.mean(loss_iter))
                
                # Evaluate model performance on the validation set
                self.mdl.eval()
                
                # Disable teacher forcing for validation
                self.mdl.teacher_forcing_ratio = 0
                
                batch_gen = func_batcher(self.batch_size,
                                         self.points_per_block - self.total_steps
                                         )
                loss_iter = []
                
                with torch.no_grad():
                    # Iterate through all the validation batches in this epoch
                    for bat in batch_gen:
                        x_bat, y_bat = x_val[bat, :, :], y_val[bat, :]
                        preds_val = (self.mdl.forward(x_bat, y_bat)).to(device)
                        rmse_val, _ = self.func_loss(preds_val, y_bat.view((self.batch_size, self.output_steps, 1)), self.master_vals[0, :])
                        loss_iter.append(rmse_val.tolist())
                        
                    self.loss_val.append(np.mean(loss_iter))
                    print(f'Training block {block + 1}/{self.num_training_blocks}, iter {i + 1}:\t Training loss: {self.loss_trn[-1]:.4f}\t Validation loss: {self.loss_val[-1]:.4f}')
                
                # Increment total_iters by 1 (used for performance plotting)
                self.total_iters += 1
                
                # Find the minimum validation loss in this block
                block_min_val = np.min(self.loss_val[-(i + 1):])
                
                # If the last iteration had the minimum validation loss,
                # save it as self.best_mdl
                if block_min_val == self.loss_val[-1]:
                    self.best_mdl = self.mdl
                    
                # Check to see if max_iters_increasing_validation_loss has
                # been exceeded (i.e. the minimum validation loss in the block
                # must have occured in the last that many iterations)
                if i >= max_iters_increasing_validation_loss:
                    recent_vals = np.array(self.loss_val[-max_iters_increasing_validation_loss:])
                    
                    # If the limit has been exceeded, exit the training loop
                    # for this block
                    if (recent_vals > block_min_val).all():
                        break
                    
                    # It's possible that the model was initialized to an area
                    # of zero gradient - in this case, exit the entire training loop
                    if (block == 0) & (len(set(self.loss_trn)) == 1):
                        self.stuck_in_zero_grad = True
                        break
            
            # Exit loop if in area of zero gradient
            if self.stuck_in_zero_grad:
                break
            
            # Load self.best_mdl (lowest validation loss on final iteration)
            # as self.mdl and compute test set performance
            self.mdl = self.best_mdl
            self.test(block)
        
        # Record the final training time and rmse
        self.train_time = (time.perf_counter() - start_time)/60
        if self.blocked_crossval == True:
            self.rmse_final = np.average(self.loss_tst[:, 1])
        else:
            self.rmse_final = self.loss_tst[-1, 1]
        print(f'\n\nTraining complete. Training time = {self.train_time:.1f} minutes.')
        print(f'\nAverage model performance on test sets: RMSE = {self.rmse_final:.4f}')
    
    # Test the model
    def test(self, block):
        # Load testing data for the current block
        xy_tst = self.xy[block + 2, :, :, :].view((self.points_per_block,
                                                   self.total_steps,
                                                   self.total_dims)
                                                  )
                 
        # Remove the first total_steps examples to avoid data leakage
        x_tst = (xy_tst[self.total_steps:, :-self.output_steps, :]).to(device)
        # y_tst only needs effluent turbidity as teacher forcing is not used
        y_tst = (xy_tst[self.total_steps:, -self.output_steps:, 0]).to(device)
        
        # Create a "batch generator" that will load all the batches for
        # this iteration (epoch)
        batch_gen = func_batcher(self.batch_size,
                                 self.points_per_block - self.total_steps
                                 )
        
        # Empty the testing loss list
        loss_tst = []
        
        with torch.no_grad():
            # Iterate through all the validation batches in this epoch
            for bat in batch_gen:
                x_bat, y_bat = x_tst[bat, :, :], y_tst[bat, :]
                preds_tst = (self.mdl.forward(x_bat, y_bat)).to(device)
                rmse_tst, mae_tst = self.func_loss(preds_tst, y_bat.view((self.batch_size, self.output_steps, 1)), self.master_vals[0, :], testing=True)
                loss_tst.append(rmse_tst.tolist())
                self.residuals_tst.append(mae_tst.tolist())
        
            # Save the average loss for this iteration to the testing loss array
            rmse_tst = np.mean(loss_tst)
            self.loss_tst[block, :] = (self.total_iters, rmse_tst)
            print(f'Model performance on test set (block {block + 3}): RMSE = {rmse_tst:.4f}')
    
    
    # Function to calculate loss in NTU
    def func_loss(self, preds, actual, master, testing=False):
        loss = nn.MSELoss()
        mae = None
        # If testing we also want MAE loss, otherwise only MSE
        if testing: loss_MAE = nn.L1Loss(reduction='none')
        # Get un-normalized values
        preds_real = func_unnormalize_variables(preds, master)
        actual_real = func_unnormalize_variables(actual, master)
        # Compute RMSE in original units (real space - NTU in the case of turbidity))
        rmse = torch.sqrt(loss(preds_real, actual_real)).to(device)
        if testing: mae = loss_MAE(preds_real, actual_real)
        return rmse, mae
    
    
    # Save the model - this includes:
    #   - A .pt file of its parameters
    #   - A .txt file logging the characteristics of the model, its performance,
    #       and the characteristics of the optimizer
    #   - A .npy file of the residuals (MAE) for all points in each test block
    #   - A .npy file of the training performance (RMSE) values for each epoch
    #   - A .png file of a plot of the training performance
    def save(self):
        if self.blocked_crossval:
            modelname = (f"{self.rnn_type}_BCV_"
                         f"input-{self.total_steps}x{self.total_dims}_"
                         f"hidden-{self.num_layers}x{self.hidden_dims}_"
                         f"RMSE-{self.rmse_final:.4f}"
                         )
        else:
            modelname = (f"{self.rnn_type}_HO_"
                         f"input-{self.total_steps}x{self.total_dims}_"
                         f"hidden-{self.num_layers}x{self.hidden_dims}_"
                         f"RMSE-{self.rmse_final:.4f}"
                         )
        
        # Save the model parameters as a .pt file
        torch.save(self.mdl, f"{self.f_model}{modelname}.pt")
        
        # Save a log detailing the characteristics of the model, its performance,
        # and the characteristics of the optimizer as a .txt file
        with open(f"{self.f_model}{modelname}.txt", 'w') as f:
            f.write(modelname)
            f.write(f"\nData set: {self.f_data}{self.training_data_filename}")
            f.write(f"\nOutput model: {self.f_model}{modelname}.pt")
            f.write("\n\nModel hyperparameters:")
            f.write(f"\nType: {self.rnn_type}")
            for param_name, param in zip(self.mdl_parameter_names, self.mdl_parameters):
                f.write(f"\n{param_name}: {param}")
            f.write(f"\n\nTraining time = {self.train_time:.1f} minutes on a {torch.cuda.get_device_name()} over {self.total_iters} iterations.")
            f.write(f"\nFinal model performance on test set: RMSE = {self.rmse_final:.4f}")
            f.write("\n\nOptimizer specifications:")
            for i in self.opt.defaults.items():
                item = str(i[0])
                value = self.opt.state_dict()['param_groups'][0][item]
                f.write(f"\n{item}: {value}")
        
        # Save the test residuals as a .txt file
        np.save(f"{self.f_model}{modelname}_residuals", np.array(self.residuals_tst).reshape((-1, self.output_steps)))
        
        # Save a plot of the training performance as a .png file
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1, self.total_iters + 1), self.loss_trn, 'r-')
        plt.plot(range(1, self.total_iters + 1), self.loss_val, 'g-')
        if self.blocked_crossval == True:
            plt.plot(self.loss_tst[:,0], self.loss_tst[:,1], 'bo', fillstyle='full')
            plt.plot([0, self.total_iters], [self.rmse_final, self.rmse_final], 'b-')
            for i in range(len(self.loss_tst)):
                plt.text(self.loss_tst[i,0] + 1, self.loss_tst[i,1], i + 3, fontsize=10)
        else:
            plt.plot(self.loss_tst[-1,0], self.loss_tst[-1,1], 'bo', fillstyle='full')
            plt.plot([0, self.total_iters], [self.rmse_final, self.rmse_final], 'b-')
            plt.text(self.loss_tst[-1,0] + 5, self.loss_tst[-1,1], len(self.loss_tst) + 2, fontsize=10)
        plt.title(modelname)
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend(['Training', 'Validation', 'Test (block)', 'Test (mean)'])
        fig.set_dpi(100.)
        plt.savefig(f"{self.f_model}{modelname}.png")
        
        # Save the training performance data as a .npy file
        np.save(f"{self.f_model}{modelname}_perf",
                np.array((self.total_iters + 1,
                          self.loss_trn,
                          self.loss_val,
                          self.loss_tst,
                          self.rmse_final),
                         dtype=object),
                allow_pickle=True
                )
        
#%% MAIN

if __name__ == '__main__':
    # You can put multiple combinations of rnn type, number of layers,
    # and hidden layer size and automatically iterate through all
    # of them
    for algo, layer, hidden in [("GRU", 2, 32), ("GRU", 3, 32), ("GRU", 4, 32),
                                ("LSTM", 2, 32), ("LSTM", 3, 32), ("LSTM", 4, 32)]:
            trainer = rnn_trainer(algo, f_data, f_model)
            trainer.initialize(
                               training_data_filename="TRAIN_1707-2009_45x7_n.pt",
                               master_values_filename="norm_values_2401-2401_45x7.pt",
                               batch_size=64,
                               output_steps=15,
                               num_training_blocks=12,
                               hidden_dims=hidden,
                               num_layers=layer,
                               teacher_forcing_ratio=0.5,
                               )
            trainer.train(
                          learning_rate=5e-4,
                          max_iters_per_block=100,
                          max_iters_increasing_validation_loss=10,
                          weight_decay=1e-8,
                          gradient_clipping_norm=1,
                          blocked_crossval=True
                          )
            trainer.save()