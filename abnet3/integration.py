#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script contains different integration units, which receive
multiple inputs and produce batches used for training
"""

from torch import cat, zeros
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

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

        concat_batch = cat(x_list, 1)

        return concat_batch

    def forward(self, x1_list, x2_list, y):
        X1_batch = self.integration_method(x1_list)
        X2_batch = self.integration_method(x2_list)
        return X1_batch, X2_batch, y

class MultitaskIntegration(IntegrationUnitBuilder):
    """
    Specify parameters and description
    """

    def __init__(self, representation_modes, feed_modes, *args, **kwargs):
        super(MultitaskIntegration, self).__init__(*args, **kwargs)

        self.rep_modes = representation_modes
        self.feed_modes = feed_modes

        #TODO: different probebilities per method

    def apply_mode_mask(self, mode_map, features):
        """
        Receives features and mode map and returns the new vector

        :param mode_map:    map for the new vector, binary vector of the same
                            dimension as the number of features, on which every
                            dimension must be 1 for the feature to appear on
                            the mapped vector, and 0 if it must be zeroed out
        :param features:    list of features, which order must correspond to the
                            mode_map one.

        :returns mapped_vector: with dimension equal to the sum of the input
                                features dimensions

        :example: for mode map [0, 1], the first input will be zeroed out and the
                  second one will show on the final vector
        """

        #TODO: maybe there's a more efficient way to do this

        assert len(mode_map) == len(features), "Mode map incongruent with features list"

        to_cat = []
        for i in range(len(mode_map)):
            if mode_map[i]:
                to_cat.append(features[i])
            else:
                to_cat.append(Variable(zeros(features[i].size())))

        mapped_vector = cat(to_cat)
        return mapped_vector

    def integration_method(self, x1_list, x2_list):
        num_pairs = len(x1_list[0])
        x1_zipped = list(zip(*x1_list))
        x2_zipped = list(zip(*x2_list))

        x1_to_cat = []
        x2_to_cat = []

        for i in range(num_pairs):
            pair_mode = np.random.choice(self.feed_modes)
            X1 = self.apply_mode_mask(self.rep_modes[pair_mode[0]], x1_zipped[i])
            X2 = self.apply_mode_mask(self.rep_modes[pair_mode[1]], x2_zipped[i])

            x1_to_cat.append(X1)
            x2_to_cat.append(X2)

        X1_batch = cat(x1_to_cat)
        X2_batch = cat(x2_to_cat)
        return X1_batch, X2_batch

    def forward(self, x1_list, x2_list, y):
        X1_batch, X2_batch = self.integration_method(x1_list, x2_list)
        return X1_batch, X2_batch, y
