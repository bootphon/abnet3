#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script contains different integration units, which receive
multiple inputs and produce batches used for training
"""

import torch
import torch.nn as nn

class IntegrationUnitBuilder(nn.Module):

    """
    Base class for integration units
    """

    def __init__(self, cuda, *args, **kwargs):
        super(IntegrationUnitBuilder, self).__init__()

        self.cuda = cuda

    def integration_method(self, *args, **kwargs):
        raise NotImplementedError('Unimplemented integration_method for class:',
                                  self.__class__.__name__)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Unimplemented forward for class:',
                                  self.__class__.__name__)

    def whoami(self, *args, **kwargs):
        """Output description for the neural network and all parameters

        """
        raise NotImplementedError('Unimplemented whoami for class:',
                                  self.__class__.__name__)

class ConcatenationIntegration(IntegrationUnitBuilder):

    def __init__(self, *args, **kwargs):
        super(ConcatenationIntegration, self).__init__(*args, **kwargs)

    def integration_method(self, x_list):
        """
        Receives batch list of inputs and concatenates them

        :param x_list: Batch list of inputs that should have the same
                       number of rows (dimension 0)

        """

        print("Concatenating: ")
        i = 1
        for input_mode in x_list:
            print("Input {} with size {}".format(i, input_mode.size()))
            i += 1
        print()
        concat_batch = torch.cat(x_list, 1)

        return concat_batch

    def forward(self, x1_list, x2_list, y):
        X1_batch = self.integration_method(x1_list)
        X2_batch = self.integration_method(x2_list)
        return X1_batch, X2_batch, y
