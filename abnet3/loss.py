#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script is composed of the different loss functions implemented for
ABnet3 based on the Autograd of Pytorch.

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
        return {'params': self.__dict__, 'class_name': self.__class__.__name__}


class coscos2(LossBuilder):
    """coscos2 Loss function

    """

    def __init__(self, avg=True, *args, **kwargs):
        super(coscos2, self).__init__(*args, **kwargs)
        self.avg = avg

    def forward(self, input1, input2, y):
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
        cos_sim = cos(input1, input2)
        idx = torch.eq(y, 1)
        cos_sim[idx] = (1-cos_sim[idx])/2
        idx = torch.eq(y, -1)
        cos_sim[idx] = torch.pow(cos_sim[idx], 2)
        output = cos_sim.sum()
        if self.avg:
            output = torch.div(output, input1.size()[0])
        return output


class cosmargin(LossBuilder):
    """cosmargin Loss function

    Parameters
    ----------
    margin : float variable between 0 and 1

    """

    def __init__(self, avg=True, margin=0.5, *args, **kwargs):
        super(cosmargin, self).__init__(*args, **kwargs)
        self.margin = margin
        self.avg = avg
        assert (margin >= 0 and margin <= 1)

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
        cos_sim = cos(input1, input2)
        idx = torch.eq(y, 1)
        cos_sim[idx] = -cos_sim[idx]
        idx = torch.eq(y, -1)
        cos_sim[idx] = torch.clamp(cos_sim[idx]-self.margin, min=0)
        output = cos_sim.sum()
        if self.avg:
            output = torch.div(output, input1.size()[0])
        return output


class weighted_loss_multi(LossBuilder):
    """weighted_loss_multi Loss function
    This a weighted loss for multi-task training based on another loss

    Parameters
    ----------
    loss : abnet3.loss functions
        loss for both tasks
    weight : float
        variable between 0 and 1, to weight one or the other task.

    """

    def __init__(self, avg=True, loss_phn=None, loss_spk=None,
                 weight=0.5, *args, **kwargs):
        super(weighted_loss_multi, self).__init__(*args, **kwargs)
        assert type(weight) is float
        assert (weight >= 0 and weight <= 1)
        # assert loss_phn in (coscos2, cosmargin), 'basis loss not implemented'
        # assert loss_spk in (coscos2, cosmargin), 'basis loss not implemented'
        self.weight = weight
        self.avg = avg
        self.loss_phn = loss_phn
        self.loss_spk = loss_spk

    def forward(self, emb_spk1, emb_phn1, emb_spk2, emb_phn2,
                y_spk, y_phn):
        """Return loss value coscos2_weighted_multi for a batch

        Parameters
        ----------
        input1, input2 : Pytorch Variable
            Input continuous vectors
        y_spk : Pytorch Variable
            Labels for input speakers
        y_phn : Pytorch Variable
            Labels for input phones
        """

        output_spk = self.loss_spk(emb_spk1, emb_spk2, y_spk)
        output_phn = self.loss_phn(emb_phn1, emb_phn2, y_phn)
        output = self.weight*output_spk + (1.0-self.weight)*output_phn
        return output


if __name__ == '__main__':

    N_batch = 16
    x1 = Variable(torch.randn(N_batch, 10))
    x2 = Variable(torch.randn(N_batch, 10))
    y = Variable(torch.from_numpy(np.random.choice([1, -1], N_batch)))
    loss = cosmargin()
    res = loss(x1, x2, y)
#    output = sia.forward_once(x)
