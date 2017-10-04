#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:22:41 2017

@author: Rachid Riad
"""



import torch
import torch.nn as nn

class LossBuilder(nn.Module):
    '''Generic Loss function for ABnet3'''
    
    def __init__(self):
        super(LossBuilder, self).__init__()
        
    def forward(self,  *args, **kwargs):
        '''Compute Loss function'''
        raise NotImplementedError('Unimplemented forward for class:',
                                  self.__class__.__name__)
    
    def whoami(self, *args, **kwargs):
        '''Output description for the loss function'''
        return self.__class__.__name__
    
    
class coscos2(LossBuilder):
    '''coscos2 Loss function'''
    
    def __init__(self):
        super(coscos2, self).__init__()
    
    def forward(self, input1, input2, y, avg=True):
        cos = nn.CosineSimilarity(dim=3, eps=1e-6)
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
    '''cosmargin Loss function'''
    
    def __init__(self, margin=0.5):
        super(cosmargin, self).__init__()
        self.margin = margin
    
    def forward(self, input1, input2, y, avg=True):
        cos = nn.CosineSimilarity(dim=3, eps=1e-6)
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
#if __name__ == '__main__':
#    
##    x = Variable(torch.randn(N_batch, 1, 1, 3))
##    output = sia.forward_once(x)
#
