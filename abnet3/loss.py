#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:22:41 2017

@author: Rachid Riad
"""



import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class LossBuilder(nn.Module):
    """Generic Loss class for ABnet3
    
    """
    
    def __init__(self, *args, **kwargs):
        super(LossBuilder, self).__init__(*args, **kwargs)
        
    def forward(self,  *args, **kwargs):
        """Compute Loss function
        
        """
        raise NotImplementedError('Unimplemented forward for class:',
                                  self.__class__.__name__)
    
    def whoami(self, *args, **kwargs):
        """Output description for the loss function
        
        """
        return {'params':self.__dict__,'class_name': self.__class__.__name__} 
    
    
class coscos2(LossBuilder):
    """coscos2 Loss function
    
    """
    
    def __init__(self, *args, **kwargs):
        super(coscos2, self).__init__(*args, **kwargs)
    
    def forward(self, input1, input2, y, avg=True):
        """Return loss value coscos2 for a batch
        
        Parameters
        ----------
        input1, input2 : Pytorch Variable
            Input continuous vectors
        y : Pytorch Variable
            Labels for inputs
        """

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        assert input1.size() == input2.size(),  'Input not the same size'
        cos_sim = cos(input1,input2)
        idx = torch.eq(y,1)
        cos_sim[idx] = (1-cos_sim[idx] )/2
        idx = torch.eq(y,-1)
        cos_sim[idx] = torch.pow(cos_sim[idx],2)
        output = cos_sim.sum()
        if avg:
            output = torch.div(output,input1.size()[0])
        return output
    
class cosmargin(LossBuilder):
    """cosmargin Loss function
    
    
    """
    
    def __init__(self, margin=0.5, *args, **kwargs):
        super(cosmargin, self).__init__(*args, **kwargs)
        self.margin = margin
    
    def forward(self, input1, input2, y, avg=True):
        """Return loss value cos margin for a batch
        
        Parameters
        ----------
        input1, input2 : Pytorch Variable
            Input continuous vectors
        y : Pytorch Variable
            Labels for inputs
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        assert input1.size() == input2.size(), 'Input not the same size'
        cos_sim = cos(input1,input2)
        idx = torch.eq(y,1)
        cos_sim[idx] = -cos_sim[idx] 
        idx = torch.eq(y,-1)
        cos_sim[idx] = torch.clamp(cos_sim[idx]-self.margin,0)
        output = cos_sim.sum()
        if avg:
            output = torch.div(output,input1.size()[0])
        return output
        
        
#
if __name__ == '__main__':
    
    N_batch = 16
    x1 = Variable(torch.randn(N_batch, 10))
    x2 = Variable(torch.randn(N_batch, 10))
    y = Variable(torch.from_numpy(np.random.choice([1,-1],N_batch)))
    loss = coscos2()
    res = loss.forward(x1,x2,y)
#    output = sia.forward_once(x)

