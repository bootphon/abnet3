#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script contains different integration units, which receive
multiple inputs and produce batches used for training
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

activation_functions = {'relu': nn.ReLU(),
                        'sigmoid': nn.Sigmoid(),
                        'tanh': nn.Tanh()}

init_functions = {'xavier_uni': nn.init.xavier_uniform,
                  'xavier_normal': nn.init.xavier_normal,
                  'orthogonal': torch.nn.init.orthogonal}

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

    @staticmethod
    def integration_method(x_list):
        """
        Receives batch list of inputs and concatenates them

        :param x_list: Batch list of inputs that should have the same
                       number of rows (dimension 0)

        """

        concat_batch = torch.cat(x_list, 1)

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

        #TODO: different probabilities per method

    @staticmethod
    def apply_mode_mask(mode_map, features):
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
                to_cat.append(Variable(torch.zeros(features[i].size())))

        mapped_vector = torch.cat(to_cat)
        return mapped_vector

    def integration_method(self, x1_list, x2_list):
        num_pairs = len(x1_list[0])
        x1_zipped = list(zip(*x1_list))
        x2_zipped = list(zip(*x2_list))

        x1_to_cat = []
        x2_to_cat = []

        for i in range(num_pairs):
            mode_idx = np.random.randint(len(self.feed_modes))
            pair_mode = self.feed_modes[mode_idx]
            X1 = self.apply_mode_mask(self.rep_modes[pair_mode[0]], x1_zipped[i])
            X2 = self.apply_mode_mask(self.rep_modes[pair_mode[1]], x2_zipped[i])

            x1_to_cat.append(X1)
            x2_to_cat.append(X2)

        X1_batch = torch.stack(x1_to_cat, 0)
        X2_batch = torch.stack(x2_to_cat, 0)
        return X1_batch, X2_batch

    def forward(self, x1_list, x2_list, y):
        X1_batch, X2_batch = self.integration_method(x1_list, x2_list)
        return X1_batch, X2_batch, y

class BiWeightedIntegration(IntegrationUnitBuilder):
    """
    Specify parameters and description
    """

    def __init__(self, activation_type, init_type, *args, **kwargs):
        super(BiWeighted, self).__init__(*args, **kwargs)
        assert activation_type in ('relu', 'sigmoid', 'tanh')
        assert init_type in ('xavier_uni', 'xavier_normal', 'orthogonal')

        self.activation = activation_functions[activation_type]
        self.activation_type = activation_type
        self.init_function = init_functions[init_type]
        self.padding_size = 0
        self.shorter_input = None
        self.constructed = False

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = init_functions[self.init_function]
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_type))
            layer.bias.data.fill_(0.0)

    def construct_linear_projection(self, input_dim, output_dim):
        projection_layer = [nn.Linear(input_dim, output_dim),
                            self.activation]
        return nn.Sequential(*projection_layer)

    def init_net(self, features):
        dim1 = features[0].size()
        dim2 = features[1].size()
        attention_vector_dim = max(dim1, dim2)


        self.linear1 = self.construct_linear_projection(dim1, attention_vector_dim)
        self.linear2 = self.construct_linear_projection(dim2, attention_vector_dim)
        self.apply(self.init_weight_method)

        self.padding_size = abs(dim1 - dim2)
        if dim1 < dim2:
            self.shorter_input = 0
        elif dim2 < dim1:
            self.shorter_input = 1


    def compute_attention_vector(self, i1, i2):
        linear1_output = self.linear1(i1)
        linear2_output = self.linear2(i2)

        return torch.add(net1_output, value=1, net2_output)

    def integration_method(self, i1, i2):
        attention_vector = self.compute_attention_vector(i1, i2)
        attention_complement = torch.add(torch.mul(attention_vector, -1), 1) # (1 - attention)

        #TODO: use padding pytorch function
        if self.shorter_input == 0:
            padding = Variable(torch.zeros(self.padding_size))
            i1 = torch.cat([i1, padding], 0)
        elif self.shorter_input == 1:
            padding = Variable(torch.zeros(self.padding_size))
            i2 = torch.cat([i2, padding], 0)

        term1 = torch.mul(attention_vector, i1)
        term2 = torch.mul(attention_complement, i2)

        return torch.add(term1, value=1, term2)


    def forward(self, x1_list, x2_list, y):
        num_pairs = len(x1_list[0])
        x1_zipped = list(zip(*x1_list))
        x2_zipped = list(zip(*x2_list))

        x1_to_cat = []
        x2_to_cat = []

        for i in range(num_pairs):
            X1 = self.integration_method(*x1_zipped[i])
            X2 = self.integration_method(*x2_zipped[i])

            x1_to_cat.append(X1)
            x2_to_cat.append(X2)

        X1_batch = torch.stack(x1_to_cat, 0)
        X2_batch = torch.stack(x2_to_cat, 0)
        return X1_batch, X2_batch, y
