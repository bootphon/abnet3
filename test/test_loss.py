#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:24:47 2017

@author: Rachid Riad
"""

import pytest
from abnet3.loss import coscos2, cosmargin
import torch
from torch.autograd import Variable
import numpy as np

losses = {
    'coscos2': coscos2,
    'cosmargin': cosmargin
    }

params = [a for a in losses ]

@pytest.mark.parametrize('loss_func,', params)
def test_forward(loss_func):
    N_batch = 16
    x1 = Variable(torch.randn(N_batch, 10))
    x2 = Variable(torch.randn(N_batch, 10))
    y = Variable(torch.from_numpy(np.random.choice([1,-1],N_batch)))
    loss = losses[loss_func]()
    res = loss(x1,x2,y)
    assert res.dim() == 1, 'fail for {}'.format(loss_func)

