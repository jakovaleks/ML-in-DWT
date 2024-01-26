# This file contains model classes for all models.

import torch
import random
import lmu
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# Set the model type based on the string entered into the trainer class
def set_model_type(model_type, mdl_parameters):
    return eval(model_type + str(mdl_parameters))

# =========================================================================== #
### ====== LSTM ===== ###
class LSTM(nn.Module):
    # Initialize an encoder-decoder LSTM based on the given hyperparameters
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size,
                 input_steps,
                 output_steps,
                 teacher_forcing_ratio,
                 ):
        super(LSTM, self).__init__()
        
        # Initialize the encoder
        self.encoder = LSTM_Encoder(total_dims,
                                    hidden_dims,
                                    num_layers,
                                    batch_size
                                    )
        
        # Initialize the decoder
        self.decoder = LSTM_Decoder(total_dims,
                                    hidden_dims,
                                    num_layers
                                    )
        
        # Save required hyperparameters
        self.total_dims = total_dims
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Add a model description to print to the console
        self.desc = f'encoder-decoder LSTM {total_dims}-{num_layers}x{hidden_dims}-1'
    
    # Forward pass on the data
    def forward(self, data_in, true_out):
        # Create empty predictions tensor
        preds = torch.zeros(self.batch_size, self.output_steps, 1).to(device)
        
        # Run forward pass on encoder
        h1, c1 = self.encoder.forward(data_in)
        
        # Take last time step of the input as first input to the decoder
        x0 = data_in[:, -1, :].view((self.batch_size, 1, self.total_dims))
        
        # Loop through the decoder for the desired number of output steps
        for t in range(self.output_steps):
            # Run forward pass on the decoder for one time step
            data_out, pred, h1, c1 = self.decoder.forward(x0, h1, c1)
            
            # Add the predicted effluent turbidity to the predictions tensor
            preds[:, t, :] = pred
            
            # Roll to see if we will use teacher forcing for the next time step
            teacher_forcing_roll = (random.random() < self.teacher_forcing_ratio)
            
            # Set the next input to the decoder (either the previous prediction
            # or the ground truth, depending on teacher_forcing_roll)
            if teacher_forcing_roll:
                x0 = true_out[:, t, :].view((self.batch_size, 1, self.total_dims))
            else:
                x0 = data_out
        
        # Return predicted effluent turbidity
        return preds

# Encoder module for LSTM
class LSTM_Encoder(nn.Module):
    # Initialize the encoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size
                 ):
        super(LSTM_Encoder, self).__init__()
        
        # Encoder LSTM layer
        self.lstm = nn.LSTM(input_size=total_dims,
                            hidden_size=hidden_dims,
                            num_layers=num_layers, 
                            batch_first=True,
                            device=device
                            )

    # Encoder forward pass on the data
    def forward(self, data_in):
        
        # Pass through LSTM layer
        data_out, (h1, c1) = self.lstm(data_in)
        
        # Return hidden and cell state 
        return h1, c1

# Decoder module for LSTM
class LSTM_Decoder(nn.Module):
    # Initialize the decoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers
                 ):
        super(LSTM_Decoder, self).__init__()
        
        # Decoder LSTM layer
        self.lstm = nn.LSTM(input_size=total_dims,
                            hidden_size=hidden_dims,
                            num_layers=num_layers, 
                            batch_first=True,
                            device=device
                            )
        
        # Decoder output layer
        self.affine = nn.Linear(hidden_dims, total_dims, device=device)
     
    # Decoder forward pass on the data
    def forward(self, data_in, h1, c1):

        # Pass through LSTM layers using hidden and cell states from the encoder        
        data_out, (h1, c1) = self.lstm(data_in, (h1, c1))
        
        # Pass through output layer
        data_out = self.affine(data_out)
        
        # Return only the effluent turbidity (column 0) as a prediction
        pred = data_out[:, :, 0]
        
        # Return the output, prediction, and hidden and cell states
        return data_out, pred, h1, c1

# =========================================================================== #


# =========================================================================== #
### ====== GRU ===== ###
class GRU(nn.Module):
    # Initialize an encoder-decoder GRU based on the given hyperparameters
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size,
                 input_steps,
                 output_steps,
                 teacher_forcing_ratio,
                 ):
        super(GRU, self).__init__()
        
        # Initialize the encoder
        self.encoder = GRU_Encoder(total_dims,
                                   hidden_dims,
                                   num_layers,
                                   batch_size
                                   )
        
        # Initialize the decoder
        self.decoder = GRU_Decoder(total_dims,
                                   hidden_dims,
                                   num_layers
                                   )
        
        # Save required hyperparameters
        self.total_dims = total_dims
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Add a model description to print to the console
        self.desc = f'encoder-decoder GRU {total_dims}-{num_layers}x{hidden_dims}-1'
    
    # Forward pass on the data
    def forward(self, data_in, true_out):
        # Create empty predictions tensor
        preds = torch.zeros(self.batch_size, self.output_steps, 1).to(device)
        
        # Run forward pass on encoder
        h1 = self.encoder.forward(data_in)
        
        # Take last time step of the input as first input to the decoder
        x0 = data_in[:, -1, :].view((self.batch_size, 1, self.total_dims))
        
        # Loop through the decoder for the desired number of output steps
        for t in range(self.output_steps):
            # Run forward pass on the decoder for one time step
            data_out, pred, h1 = self.decoder.forward(x0, h1)
            
            # Add the predicted effluent turbidity to the predictions tensor
            preds[:, t, :] = pred
            
            # Roll to see if we will use teacher forcing for the next time step
            teacher_forcing_roll = (random.random() < self.teacher_forcing_ratio)
            
            # Set the next input to the decoder (either the previous prediction
            # or the ground truth, depending on teacher_forcing_roll)
            if teacher_forcing_roll:
                x0 = true_out[:, t, :].view((self.batch_size, 1, self.total_dims))
            else:
                x0 = data_out
        
        # Return predicted effluent turbidity
        return preds

# Encoder module for GRU
class GRU_Encoder(nn.Module):
    # Initialize the encoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size,
                 ):
        super(GRU_Encoder, self).__init__()
        
        # Encoder GRU layer
        self.gru = nn.GRU(input_size=total_dims,
                          hidden_size=hidden_dims,
                          num_layers=num_layers, 
                          batch_first=True,
                          device=device
                          )

    # Encoder forward pass on the data
    def forward(self, data_in):
        
        # Pass through GRU layer
        data_out, h1 = self.gru(data_in)
        
        # Return hidden state 
        return h1

# Decoder module for GRU
class GRU_Decoder(nn.Module):
    # Initialize the decoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers
                 ):
        super(GRU_Decoder, self).__init__()
        
        # Decoder GRU layer
        self.gru = nn.GRU(input_size=total_dims,
                          hidden_size=hidden_dims,
                          num_layers=num_layers, 
                          batch_first=True,
                          device=device
                          )
        
        # Decoder output layer
        self.affine = nn.Linear(hidden_dims, total_dims, device=device)
     
    # Decoder forward pass on the data
    def forward(self, data_in, h1):

        # Pass through GRU layers using hidden and cell states from the encoder        
        data_out, h1 = self.gru(data_in, h1)
        
        # Pass through output layer
        data_out = self.affine(data_out)
        
        # Return only the effluent turbidity (column 0) as a prediction
        pred = data_out[:, :, 0]
        
        # Return the output, prediction, and hidden and cell states
        return data_out, pred, h1

# =========================================================================== #


# =========================================================================== #
### ====== MLP ===== ###
class MLP(nn.Module):
    # Initialize the decoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size,
                 input_steps,
                 output_steps,
                 activation_type
                 ):
        super(MLP, self).__init__()

        activations_list = ['Sigmoid', 'Tanh', 'ReLU']
        assert activation_type in activations_list, f'activation_type must be one of {activations_list}'
        # Note: the last layer activation is always ReLU, since turbidity is unbounded
        # Some trial-and-error suggests that ReLU is the best activation
        # function for all layers
        self.activation_type = activation_type
        
        # Stack all the layers
        self.mlp = nn.Sequential(nn.Linear(total_dims * input_steps, hidden_dims * input_steps))
        self.func_add_activation_layer()
        if num_layers > 2:
            for layers in range(num_layers - 2):
                self.mlp.append(nn.Linear(hidden_dims * input_steps, hidden_dims * input_steps))
                self.func_add_activation_layer()
        self.mlp.append(nn.Linear(hidden_dims * input_steps, output_steps))
        self.mlp.append(nn.ReLU())
        
        # Add a model description to print to the console
        self.desc = f'MLP {total_dims*input_steps}-{num_layers}x{hidden_dims*input_steps}-{output_steps}'
            
    def func_add_activation_layer(self):
        if self.activation_type == 'Sigmoid': self.mlp.append(nn.Sigmoid())
        if self.activation_type == 'Tanh': self.mlp.append(nn.Tanh())
        if self.activation_type == 'ReLU': self.mlp.append(nn.ReLU())
        
    def forward(self, data_in):
        return self.mlp(data_in)

# =========================================================================== #