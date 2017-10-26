#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:33:00 2017

@author: Rachid Riad
"""

import pytest
from abnet3.model import SiameseNetwork
from abnet3.loss import coscos2, cosmargin
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import copy

models = {
    'siamese_relu': SiameseNetwork(input_dim=10,num_hidden_layers=2,
                                   hidden_dim=10,
                                   output_dim=40,dropout=0.1,
                                   activation_layer='relu'),
                                   
    'siamese_sig': SiameseNetwork(input_dim=10,num_hidden_layers=4,
                                  hidden_dim=10,type_init='orthogonal',
                                  output_dim=15,dropout=0.,
                                  activation_layer='sigmoid')
    }

losses = {
    'coscos2': coscos2,
    'cosmargin': cosmargin
    }

params = [(a,b) for a in models for b in losses]

@pytest.mark.parametrize('model_func,loss_func', params)
def test_update_all_weights(model_func,loss_func):
    """Test to check updates to all layers in the network with manual updates
    
    """
    N_batch = 64
    x1 = Variable(torch.randn(N_batch, 10))
    x2 = Variable(torch.randn(N_batch, 10))
    y = Variable(torch.from_numpy(np.random.choice([1],N_batch)))
    loss = losses[loss_func]()
    net = models[model_func]
    learning_rate = 0.1
    param_before = copy.deepcopy(list(net.parameters()))
    output1,output2 = net(x1,x2)
    net.zero_grad()
    res = loss(output1,output2,y)
    res.backward()
    for param in net.parameters():

        param.data -= learning_rate * param.grad.data
    param_after = net.parameters()

    for layer1, layer2 in zip(param_before,param_after):
        assert (layer1 != layer2).data.numpy().any()

    
@pytest.mark.parametrize('model_func,loss_func', params)
def test_update_all_weights_with_optim(model_func,loss_func):
    """Test to check updates to all layers in the network with optim package
    
    """
    N_batch = 64
    x1 = Variable(torch.randn(N_batch, 10))
    x2 = Variable(torch.randn(N_batch, 10))
    y = Variable(torch.from_numpy(np.random.choice([1],N_batch)))
    loss = losses[loss_func]()
    net = models[model_func]
    param_before = copy.deepcopy(list(net.parameters()))
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    output1,output2 = net(x1,x2)
    optimizer.zero_grad()
    res = loss(output1,output2,y)
    res.backward()
    optimizer.step()
    param_after = net.parameters()
    for layer1, layer2 in zip(param_before,param_after):
        assert (layer1 != layer2).data.numpy().any()
    
    