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
    Receives list of features from different modalities and joins them, zeroing
    out features when required.

    :param representation_modes:    list of representation modes. A representation
                                    mode is a tuple-like object that should
                                    have the same length as the number of modalities
                                    for the integration. Every element of the
                                    representation mode is either a 1 or a 0,
                                    and corresponds to one modality used for
                                    training, so a 1 means that the corresponding
                                    modality will appear on the returned batch,
                                    and a 0 means that said modality will be zeroed
                                    out. Finally, the features, zeroed out or not,
                                    are concatenated.
    :param feed_modes:              list of feed modes. A feed mode is a 2 dimensional
                                    tuple-like object, on which each element
                                    corresponds to one branch of the siamese network.
                                    Each element must be a number, that represents
                                    the index of a representation mode from the
                                    representation_modes parameter. This tuple
                                    represents pairs of representation modes that
                                    will be used for the training.

    :param dimensions:              list of modality dimensions, meaning the length
                                    of one vector of each modality (in the order
                                    the paths were passed to the dataloader)


    :examples:                      For an integration with 2 different modalities,
                                    the representation mode (1, 0) indicates
                                    that the first modality should be used and the
                                    second one should be zeroed out.

                                    Given the  representation modes [(1, 0), (0, 1)],
                                    the feed mode (0, 1) means that for the first
                                    branch of  the siamese network the representation
                                    mode (1, 0) will be used, and for the second branch,
                                    the representation mode (0, 1) will be used.
    """

    def __init__(self, representation_modes, feed_modes, dimensions, batch_size,
                 *args, **kwargs):
        super(MultitaskIntegration, self).__init__(*args, **kwargs)

        self.rep_modes = self.bootstrap(representation_modes, dimensions)
        self.feed_modes = feed_modes
        self.batch_size = batch_size
        #TODO: different probabilities per method

    def bootstrap(self,representation_modes, dimensions_list):
        """Constructs necessary elements for integration
        """

        print("Expanding masks for multitask integration")
        expanded_rep_modes = []
        for rep_mode in representation_modes:
            expanded = []
            for binary, dimension in zip(rep_mode, dimensions_list):
                expanded += [binary] * dimension
            expanded_rep_modes.append(expanded)
        return expanded_rep_modes


    def get_batch_masks(self):
        mask1 = []
        mask2 = []
        for i in np.random.random_integers(0, len(self.feed_modes) - 1,
                                            size = self.batch_size):

            feed_mode = self.feed_modes[i]
            mask1.append(self.rep_modes[feed_mode[0]])
            mask2.append(self.rep_modes[feed_mode[1]])

        mask1 = Variable(torch.Tensor(mask1))
        mask2 = Variable(torch.Tensor(mask2))
        return mask1, mask2


    def integration_method(self, x1_list, x2_list):

        x1_cat = torch.cat(x1_list, 1)
        x2_cat = torch.cat(x2_list, 1)

        mask1, mask2 = self.get_batch_masks()

        X1_batch = torch.mul(mask1, x1_cat)
        X2_batch = torch.mul(mask2, x2_cat)

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

    def bootstrap(self, features):
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
