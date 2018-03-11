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

    def __init__(self, output_path="", cuda_bool=False, *args, **kwargs):
        super(IntegrationUnitBuilder, self).__init__()

        self.output_path = output_path
        self.cuda_bool = cuda_bool

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

    def bootstrap(self, *args, **kwargs):
        """Used for starting the integration unit
        """

    def save(self, epoch=''):
        torch.save(self.state_dict(), self.output_path + epoch + 'integration.pth')

    def load(self, path=None):
        self.load_state_dict(torch.load(path+'integration.pth'))



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

    def forward(self, x1_list, x2_list, y, embed=False, *args, **kwargs):
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


    def get_batch_masks(self, embed):
        mask1 = []
        mask2 = []
        if embed:
            size = 1
        else:
            size = self.batch_size
        for i in np.random.random_integers(0, len(self.feed_modes) - 1,
                                            size = size):

            feed_mode = self.feed_modes[i]
            mask1.append(self.rep_modes[feed_mode[0]])
            mask2.append(self.rep_modes[feed_mode[1]])

        mask1 = Variable(torch.Tensor(mask1))
        mask2 = Variable(torch.Tensor(mask2))

        if self.cuda_bool:
            mask1 = mask1.cuda()
            mask2 = mask2.cuda()

        return mask1, mask2


    def integration_method(self, x1_list, x2_list, embed):

        x1_cat = torch.cat(x1_list, 1)
        x2_cat = torch.cat(x2_list, 1)

        mask1, mask2 = self.get_batch_masks(embed)

        X1_batch = torch.mul(mask1, x1_cat)
        X2_batch = torch.mul(mask2, x2_cat)

        return X1_batch, X2_batch

    def forward(self, x1_list, x2_list, y, embed=False, *args, **kwargs):
        X1_batch, X2_batch = self.integration_method(x1_list, x2_list, embed)
        return X1_batch, X2_batch, y

class BiWeightedFixedSum(IntegrationUnitBuilder):
    """
    Sums two vectors of the same dimension, using a weight and it's compliment
    """


    def __init__(self, weight_value = 0.5, *args, **kwargs):
        super(BiWeightedFixedSum, self).__init__(*args, **kwargs)

        assert weight_value >= 0, "weight must be possitive or 0"
        assert weight_value <= 1, "weight must be less or equal than 1"
        self.weight_value = weight_value
        self.weight_complement = 1 - weight_value

    def integration_method(self, i1, i2):
        v1_weighted = torch.mul(i1, self.weight_value)
        v2_weighted = torch.mul(i2, self.weight_complement)

        return torch.add(v1_weighted, v2_weighted)



    def forward(self, x_list):
        X = self.integration_method(*x_list)
        return X


class BiWeightedLearntSum(BiWeightedFixedSum):
    """
    Sums two vectors of the same dimension, using a weight and it's compliment.
    Said weight is learnt, using a linear projection of the input vectors
    """

    def __init__(self, input_dim, activation_type, init_type='xavier_uni',
                       *args, **kwargs):
        super(BiWeightedLearntSum, self).__init__(*args, **kwargs)
        assert activation_type in ('sigmoid', 'tanh')
        assert init_type in ('xavier_uni', 'xavier_normal', 'orthogonal')

        self.activation = activation_functions[activation_type]
        self.activation_type = activation_type
        self.init_function = init_functions[init_type]
        self.linear1 = self.construct_linear_projection(input_dim, 1)
        self.linear2 = self.construct_linear_projection(input_dim, 1)
        self.apply(self.init_weight_method)

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = self.init_function
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_type))
            layer.bias.data.fill_(0.0)

    def construct_linear_projection(self, input_dim, output_dim):
        projection_layer = [nn.Linear(input_dim, output_dim),
                            self.activation]
        return nn.Sequential(*projection_layer)


    def compute_attention_weight(self, i1, i2):
        linear1_output = self.linear1(i1)
        linear2_output = self.linear2(i2)

        return torch.add(linear1_output, linear2_output)

    def integration_method(self, i1, i2):
        self.weight_value = self.compute_attention_weight(i1, i2)
        self.weight_complement = 1 - self.a_value
        return super(BiWeightedLearntSum, self).integration_method(i1, i2)

class BiWeightedFixedCat(IntegrationUnitBuilder):
    """
    Concats two vectors of the same dimension, using a vector weight and it's compliment
    """


    def __init__(self, weight_vector=None, *args, **kwargs):
        super(BiWeightedFixedCat, self).__init__(*args, **kwargs)
        self.weight_vector = weight_vector
        self.weight_complement = torch.add(torch.mul(weight_vector, -1), 1) # (1 - vector)

    def integration_method(self, i1, i2):
        v1_weighted = torch.mul(i1, self.weight_vector)
        v2_weighted = torch.mul(i2, self.weight_complement)
        return torch.cat(v1_weighted, v2_weighted)

    def forward(self, x_list):
        X = self.integration_method(*x_list)
        return X


class BiWeightedLearntCat(BiWeightedFixedCat):
    """
    Concatenates two vectors of the same dimension, using a vector weight and it's compliment.
    Said weight is learnt, using a linear projection of the input vectors
    """

    def __init__(self, input_dim, activation_type, init_type='xavier_uni',
                       *args, **kwargs):
        super(BiWeightedLearntCat, self).__init__(*args, **kwargs)
        assert activation_type in ('sigmoid', 'tanh')
        assert init_type in ('xavier_uni', 'xavier_normal', 'orthogonal')

        self.activation = activation_functions[activation_type]
        self.activation_type = activation_type
        self.init_function = init_functions[init_type]
        self.linear1 = self.construct_linear_projection(input_dim, input_dim*2)
        self.linear2 = self.construct_linear_projection(input_dim, input_dim*2)
        self.apply(self.init_weight_method)

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = self.init_function
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_type))
            layer.bias.data.fill_(0.0)

    def construct_linear_projection(self, input_dim, output_dim):
        projection_layer = [nn.Linear(input_dim, output_dim),
                            self.activation]
        return nn.Sequential(*projection_layer)


    def compute_attention_weight(self, i1, i2):
        linear1_output = self.linear1(i1)
        linear2_output = self.linear2(i2)
        return torch.add(linear1_output, linear2_output)

    def integration_method(self, i1, i2):
        self.weight_vector = self.compute_attention_weight(i1, i2)
        self.weight_complement = torch.add(torch.mul(weight_vector, -1), 1)
        return super(BiWeightedLearntCat, self).integration_method(i1, i2)
