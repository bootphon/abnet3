#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:22:41 2017

@author: Rachid Riad
"""



import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class NetworkBuilder(nn.Module):
    """Generic Neural Network Model class
    
    """
    def __init__(self, *args, **kwargs):
        super(NetworkBuilder, self).__init__()
    
    def forward_once(self, *args, **kwargs):
        raise NotImplementedError('Unimplemented forward_once for class:',
                                  self.__class__.__name__)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError('Unimplemented forward for class:',
                                  self.__class__.__name__)
        
    def whoami(self, *args, **kwargs):
        """Output description for the neural network and all parameters
        
        """
        raise NotImplementedError('Unimplemented whoami for class:',
                          self.__class__.__name__)
    
    def save_network(self, *args, **kwargs):
        """Save network weights
        
        """
        raise NotImplementedError('Unimplemented save_network for class:',
                          self.__class__.__name__)
    
    def load_network(self, *args, **kwargs):
        """Load network weights
        
        """
        raise NotImplementedError('Unimplemented load_network for class:',
                          self.__class__.__name__)

class SiameseNetwork(NetworkBuilder):
    """Siamese neural network Architecture
    
    """
    def __init__(self, input_dim=None, num_hidden_layers=None, hidden_dim=None, 
                 output_dim=None, p_dropout=0.1, batch_norm=False,
                 activation_function=None, output_path=None, *args, **kwargs):
        super(SiameseNetwork,self).__init__()
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.batch_norm = batch_norm
        # Pass forward network functions
        self.input_emb = nn.Sequential(
                nn.Linear(input_dim,hidden_dim),
                nn.Dropout(p=p_dropout, inplace=False),
                activation_function)
        self.hidden_layer = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.Dropout(p=p_dropout, inplace=False),
                activation_function)
        self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim,output_dim),
                nn.Dropout(p=p_dropout, inplace=False),
                activation_function)
        self.output_path = output_path

        
    def forward_once(self, x):
        """Simple forward pass for one instance x
        
        """
        output = self.input_emb(x)
        for idx in range(self.num_hidden_layers):
            output = self.hidden_layer(output)
        output = self.output_layer(output)
        return output
        
    def forward(self, input1, input2):
        """Forward pass through the same network
        
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    
    def whoami(self):
        """Output description for the neural network and all parameters
        
        """
        return {'params':self.__dict__,'class_name': self.__class__.__name__}       
        
    def save_network(self):
        torch.save(self.state_dict(), self.output_path +'.nnet')
    
    def load_network(self, network_path=None):
        self.load_state_dict(torch.load(network_path))

if __name__ == '__main__':
    sia = SiameseNetwork(input_dim=3,num_hidden_layers=2,hidden_dim=10,
                         output_dim=19,dropout=0.1,
                         activation_function=nn.ReLU(inplace=True),
                         batch_norm=True)
    
    N_batch = 64
    x1 = Variable(torch.randn(N_batch, 1, 1, 3))
    x2 = Variable(torch.randn(N_batch, 1, 1, 3))
    output1, output2 = sia.forward(x1,x2)
    y = Variable(torch.LongTensor(np.random.choice([1,-1],N_batch)))























