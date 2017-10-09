#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:56:29 2017

@author: Rachid Riad
"""

import torch.nn as nn

import abnet3
from abnet3.model import *
from abnet3.loss import *
from abnet3.sampler import *
from abnet3.utils import *
from abnet3.trainer import *

import time


if __name__ == '__main__':
    
    network = SiameseNetwork(input_dim=280,num_hidden_layers=2,hidden_dim=500,
                     output_dim=100,p_dropout=0.1,
                     activation_function=nn.Sigmoid(),
                     batch_norm=False, output_path='',
                     cuda=True)
    
    sam = SamplerClusterSiamese(already_done=True, directory_output=None)
    coscos2_loss = coscos2()
    
    start_time = time.time()
    
    tra = TrainerSiamese(sam,network,coscos2_loss)
