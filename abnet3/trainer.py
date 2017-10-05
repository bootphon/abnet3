#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:24:59 2017

@author: Rachid Riad
"""

from abnet3.model import *
from abnet3.loss import *
from abnet3.sampler import *
from abnet3.utils import *

class TrainerBuilder(object):
    """Generic Trainer class for ABnet3
    
    """
    def __init__(self, sampler, network, loss, feature_path=None):
        super(TrainerBuilder, self).__init__()
        self.sampler = sampler
        self.network = network
        self.loss = loss
        self.feature_path = feature_path
        
    def whoami(self):
        return {'params':self.__dict__,
                'network':self.network.whoami(),
                'loss':self.loss.whoami(),
                'sampler':self.sampler.whoami(),
                'class_name': self.__class__.__name__}
    
    def train(self):
        """Train function 
    
        """
        raise NotImplementedError('Unimplemented train for class:',
                          self.__class__.__name__)
        
        
#class TrainerSiamese()

if __name__ == '__main__':
    
    sia = SiameseNetwork(input_dim=3,num_hidden_layers=2,hidden_dim=10,
                     output_dim=19,dropout=0.1,
                     activation_function=nn.ReLU(inplace=True),
                     batch_norm=True)
    sam = SamplerClusterSiamese()
    loss = coscos2()
    tra = TrainerBuilder(sam,sia,loss)
    
    


        
        
        