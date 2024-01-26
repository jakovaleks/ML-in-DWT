# This file is for training RNN models.

import torch
import models
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from data_torch_mgmt import func_loader, func_batcher

# File locations
f_root = "/home/user1/Documents/ML_DWT/"
f_data = "Data/Torch/"
f_model = "Models/"
fileloc_data = f"{f_root}{f_data}"
fileloc_model = f"{f_root}{f_model}"
    
# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# Class to run the training procedure
class mlp_trainer:
    def __init__(self, fileloc_data, fileloc_model):
        
        # Specify folders to load/save data/models to/from
        self.fileloc_data = fileloc_data
        self.fileloc_model = fileloc_model
    
    # Load the training data and initialize the model
    def initialize(self,
                   training_data_filename,
                   batch_size=64,
                   output_steps=15,
                   num_training_blocks=10,
                   hidden_dims=64,
                   num_layers=3,
                   activation_type='ReLU'
                   ):
        
        self.training_data_filename = training_data_filename
        self.batch_size = batch_size
        self.output_steps = output_steps
        self.num_training_blocks = num_training_blocks
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.activation_type = activation_type
        
        # Load the data as a 4D array of size (blocks, examples_per_block, steps, variables)
        self.xy = func_loader(self.fileloc_data, self.training_data_filename, self.num_training_blocks + 2)
        _, self.points_per_block, self.total_steps, self.total_dims = self.xy.shape
        self.input_steps = self.total_steps - self.output_steps
        
        # Load an unsmoothed version of the data for testing if using smoothed data for training
        if 's' in self.training_data_filename:
            unsmoothed_filename = training_data_filename.replace('s', '')
            self.unsmoothed = func_loader(self.fileloc_data, unsmoothed_filename, self.num_training_blocks + 2)
            self.x_uns = torch.reshape(self.unsmoothed[:, :, :self.input_steps, :], (self.num_training_blocks + 2, self.points_per_block, self.input_steps * self.total_dims))
            self.y_uns = torch.reshape(self.unsmoothed[:, :, self.input_steps:, 0], (self.num_training_blocks + 2, self.points_per_block, self.output_steps))
        
        self.mdl_parameters = (self.total_dims,
                               self.hidden_dims,
                               self.num_layers,
                               self.batch_size,
                               self.input_steps,
                               self.output_steps,
                               self.activation_type
                               )
        self.mdl_parameter_names = ['Total dimensions',
                                    'Hidden dimensions',
                                    'Hidden layers',
                                    'Batch size',
                                    'Input time steps',
                                    'Output time steps',
                                    'Activation type'
                                    ]
        
        # Load a model based on the above parameters
        self.mdl = models.set_model_type('MLP', self.mdl_parameters).to(device)
        
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
         
        # Flatten the data to three dimensions for MLP
        self.x = torch.reshape(self.xy[:, :, :self.input_steps, :], (self.num_training_blocks + 2, self.points_per_block, self.input_steps * self.total_dims))
        self.y = torch.reshape(self.xy[:, :, self.input_steps:, 0], (self.num_training_blocks + 2, self.points_per_block, self.output_steps))
         
        for block in training_range:
            # Load training set for the current block
            x_trn = self.x[:block + 1, :, :].view(((block + 1) * self.points_per_block,
                                                        self.input_steps * self.total_dims)
                                                        ).to(device)
            y_trn = self.y[:block + 1, :, :].view(((block + 1) * self.points_per_block,
                                                        self.output_steps)
                                                        ).to(device)
            
            # Load validation data for the current block
            x_val = self.x[block + 1, :, :].view((self.points_per_block,
                                                   self.input_steps * self.total_dims)
                                                   ).to(device)
            y_val = self.y[block + 1, :, :].view((self.points_per_block,
                                                   self.output_steps)
                                                   ).to(device)
            
            # Remove the first total_steps examples to avoid data leakage
            x_val = (x_val[self.total_steps:, :]).to(device)
            y_val = (y_val[self.total_steps:, :]).to(device)
            
            # Shuffle the order of examples in x_trn and y_trn
            trn_idx = torch.randperm((block + 1) * self.points_per_block)
            x_trn = x_trn[trn_idx]
            y_trn = y_trn[trn_idx]
                
            # Iterate training loop until either max_iters_per_block is reached,
            # or validation loss is above the lowest loss for that block for
            # max_iters_increasing_validation_loss consecutive iterations
            for i in range(max_iters_per_block):
                self.mdl.train()
                
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
                    x_bat, y_bat = x_trn[bat, :], y_trn[bat, :]
                    
                    # Perform forward pass on the data
                    preds_trn = (self.mdl.forward(x_bat)).to(device)
                    
                    # Calculate training loss on this batch
                    # We transform it back to NTU for meaningful interpretation!
                    rmse_trn, _ = self.func_loss(preds_trn, y_bat)
                                        
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
                
                batch_gen = func_batcher(self.batch_size,
                                         self.points_per_block - self.total_steps
                                         )
                loss_iter = []
                
                with torch.no_grad():
                    # Iterate through all the validation batches in this epoch
                    for bat in batch_gen:
                        x_bat, y_bat = x_val[bat, :], y_val[bat, :]
                        preds_val = (self.mdl.forward(x_bat)).to(device)
                        rmse_val, _ = self.func_loss(preds_val, y_bat)
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
        x_tst = self.x[block + 2, :, :].view((self.points_per_block,
                                               self.input_steps * self.total_dims)
                                               )
        y_tst = self.y[block + 2, :, :].view((self.points_per_block,
                                               self.output_steps)
                                               )
                 
        # Remove the first total_steps examples to avoid data leakage
        x_tst = (x_tst[self.total_steps:, :]).to(device)
        y_tst = (y_tst[self.total_steps:, :]).to(device)
        
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
                x_bat, y_bat = x_tst[bat, :], y_tst[bat, :]
                preds_tst = (self.mdl.forward(x_bat)).to(device)
                rmse_tst, mae_tst = self.func_loss(preds_tst, y_bat, testing=True)
                loss_tst.append(rmse_tst.tolist())
                self.residuals_tst.append(mae_tst.tolist())
        
            # Save the average loss for this iteration to the testing loss array
            rmse_tst = np.mean(loss_tst)
            self.loss_tst[block, :] = (self.total_iters, rmse_tst)
            print(f'Model performance on test set (block {block + 3}): RMSE = {rmse_tst:.4f}')
    
    
    # Function to calculate loss in NTU
    def func_loss(self, preds, actual, testing=False):
        loss = nn.MSELoss()
        mae = None
        if testing: loss_MAE = nn.L1Loss(reduction='none')
        # to avoid having to load the original data file every time to find
        # the maximum turbidity in the data set, simply find it once and 
        # paste it into here for future reuse
        turb_max = 10.0200002193451 
        # Since the minimum turbidity is 0, the un-normalization calculation
        # is simplified to simply multiplying the values by the maximum
        # turbidity in the original data set
        rmse = torch.sqrt(loss(turb_max * preds, turb_max * actual)).to(device)
        if testing: mae = loss_MAE(turb_max * preds, turb_max * actual)
        return rmse, mae
    
    
    # Save the model - this includes:
    #   - A .pt file of its parameters
    #   - A .txt file logging the characteristics of the model, its performance,
    #       and the characteristics of the optimizer
    #   - A .npy file of the residuals (MAE) for all points in each test block
    #   - A .npy file of the training performance (RMSE) values for each epoch
    #   - A .png file of a plot of the training performance
    def save(self):
        modelname = (f"MLP_"
                     f"input-{self.total_steps}x{self.total_dims}_"
                     f"hidden-{self.num_layers}x{self.hidden_dims}_"
                     f"{self.activation_type}_"
                     f"RMSE-{self.rmse_final:.4f}"
                     )
        
        # Save the model parameters as a .pt file
        torch.save(self.mdl, f"{self.fileloc_model}{modelname}.pt")
        
        # Save a log detailing the characteristics of the model, its performance,
        # and the characteristics of the optimizer as a .txt file
        with open(f"{self.fileloc_model}{modelname}.txt", 'w') as f:
            f.write(modelname)
            f.write(f"\nData set: {self.fileloc_data}{self.training_data_filename}")
            f.write(f"\nOutput model: {self.fileloc_model}{modelname}.pt")
            f.write("\n\nModel hyperparameters:")
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
        np.save(f"{self.fileloc_model}{modelname}_residuals", np.array(self.residuals_tst).reshape((-1, self.output_steps)))
        
        # Save a plot of the training performance as a .png file
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1, self.total_iters + 1), self.loss_trn, 'r-')
        plt.plot(range(1, self.total_iters + 1), self.loss_val, 'g-')
        if self.blocked_crossval == True:
            plt.plot(self.loss_tst[:,0], self.loss_tst[:,1], 'bo', fillstyle='full')
            plt.plot([0, self.total_iters], [self.rmse_final, self.rmse_final], 'b-')
            for i in range(len(self.loss_tst)):
                plt.text(self.loss_tst[i,0] + 5, self.loss_tst[i,1], i + 3, fontsize=10)
        else:
            plt.plot(self.loss_tst[-1,0], self.loss_tst[-1,1], 'bo', fillstyle='full')
            plt.plot([0, self.total_iters], [self.rmse_final, self.rmse_final], 'b-')
            plt.text(self.loss_tst[-1,0] + 5, self.loss_tst[-1,1], len(self.loss_tst) + 2, fontsize=10)
        plt.title(modelname)
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend(['Training', 'Validation', 'Test (block)', 'Test (mean)'])
        fig.set_dpi(100.)
        plt.savefig(f"{self.fileloc_model}{modelname}.png")
        
        # Save the training performance data as a .npy file
        np.save(f"{self.fileloc_model}{modelname}_perf",
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
    # You can put multiple combinations of hidden layer number and size
    # and automatically iterate through all of them
    for l, h in [(2, 32), (2, 64), (2, 128), (3, 32), (3, 64), (3, 128), (4, 32), (4, 64)]:
        trainer = mlp_trainer(fileloc_data, fileloc_model)
        trainer.initialize(
                           training_data_filename="TRAIN_1707-2009_45x7_n.pt",
                           batch_size=64,
                           output_steps=15,
                           num_training_blocks=12,
                           hidden_dims=h,
                           num_layers=l,
                           activation_type='ReLU',
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
            