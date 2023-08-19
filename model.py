# %% [markdown]
# Basic import of packages
# %%
import numpy as np
import torch
from torch import nn, Tensor, no_grad
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd 
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch.nn.utils.parametrize as parametrize
import math 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 

# %%
## This is the cell for RNN 
#-----------------------------RNN Module-----------------------------#
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.recurrent_kernel = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        # here you can choose your own activation function 
        out = self.nonlinearity(out)
        
        return out, out 
# %%
class LSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.c2c = Tensor(hidden_size * 3)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        #pdb.set_trace()
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        c2c = self.c2c.unsqueeze(0)
        ci, cf, co = c2c.chunk(3,1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate+ ci * cx)
        forgetgate = torch.sigmoid(forgetgate + cf * cx)
        cellgate = forgetgate*cx + ingate* torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate+ co*cellgate)
        

        hm = outgate * F.tanh(cellgate)
        return (hm, cellgate)

# %%
'''
STEP 3: CREATE MODEL CLASS
'''
 
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, model = 'lstm', bias=True):
        super(Model, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        # 
        # 
        self.model = model 
        self.layer_dim = layer_dim
        if model == 'lstm':
            self.model = LSTMCell(input_dim, hidden_dim)  
        elif model == 'rnn':
            self.model = RNNCell(input_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

                    
       
        outs = []
        
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            if self.model == 'lstm':
                hn, cn = self.model(x[:,seq,:], (hn,cn)) 
            else:
                hn, cn = self.model(x[:,seq,:], hn) 
            outs.append(hn)
            
    

        out = outs[-1].squeeze()
        
        out = self.fc(out) 
        # out.size() --> 100, 10
        return out