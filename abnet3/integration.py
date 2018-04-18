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
from abnet3.utils import expand_dimension_list

activation_functions = {'relu': nn.ReLU(),
                        'sigmoid': nn.Sigmoid(),
                        'tanh': nn.Tanh()}

init_functions = {'xavier_uni': nn.init.xavier_uniform,
                  'xavier_normal': nn.init.xavier_normal,
                  'orthogonal': torch.nn.init.orthogonal}

class IntegrationUnitBuilder(nn.Module):

    """
    Base class for integration units

    :param output_path: String, path to save the integration unit

    :param cuda_bool:   Boolean, wether cuda should be used (True) or not (False),
                        default is False
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

    def __str__(self):
        return str(self.__class__.__name__)



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

    def forward(self, x_list, *args, **kwargs):
        X = self.integration_method(x_list)
        return X

    def __str__(self):
        return str(self.__class__.__name__)

class SumIntegration(IntegrationUnitBuilder):

    def __init__(self, *args, **kwargs):
        super(SumIntegration, self).__init__(*args, **kwargs)

    @staticmethod
    def integration_method(x_list):
        """
        Receives batch list of inputs and sums them

        :param x_list: Batch list of inputs that should have the same
                       number of rows (dimension 0) and columns (dimension 1)

        """

        __sum = torch.sum(x_list[0], x_list[1])
        for _input in x_list[2:]:
            __sum = torch.sum(__sum, _input)
        return __sum

    def forward(self, x_list, *args, **kwargs):
        X = self.integration_method(x_list)
        return X

    def __str__(self):
        return str(self.__class__.__name__)

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
                                    will be used for the training. If the string
                                    'many2many' is passed, all possible combinations
                                    of representation modes will be used, and if
                                    the string 'one2one' is passed, only pairs with
                                    the same rep mode on each side will be used.
                                    All feed modes have the same probability to
                                    be chosen.

    :param dimensions:              list of modality dimensions, meaning the length
                                    of one vector of each modality (in the order
                                    the paths were passed to the dataloader)


    :examples:                      For an integration with 2 different modalities,
                                    the representation mode (1, 0) indicates
                                    that the first modality should be used and the
                                    second one should be zeroed out.

                                    Given the representation modes:
                                    [rep_tuple_1, rep_tuple_2], the feed mode
                                    (1, 0) means that for the first branch of  the
                                    siamese network the representation mode with
                                    index 1 will be used: rep_tuple_2, and
                                    for the second branch, the representation mode
                                    with index 0 will be used: rep_tuple_1.
    """

    def __init__(self, representation_modes, feed_modes, dimensions, batch_size,
                 *args, **kwargs):
        super(MultitaskIntegration, self).__init__(*args, **kwargs)
        self.rep_modes = []
        self.feed_modes = feed_modes
        self.batch_size = batch_size
        self.unexpanded_rep_modes = representation_modes

        self.next_mask = None

        self.bootstrap(representation_modes, dimensions)

    def bootstrap(self, representation_modes, dimensions_list):
        """Constructs necessary elements for integration
        """

        print("Expanding masks for multitask integration")
        for rep_mode in representation_modes:
            expanded = []
            for binary, dimension in zip(rep_mode, dimensions_list):
                expanded += [binary] * dimension
            self.rep_modes.append(expanded)

        if self.feed_modes == "many2many":
            print("Creating feed modes, many2many")
            self.feed_modes = []
            for i in range(len(self.rep_modes)):
                for j in range(len(self.rep_modes)):
                    self.feed_modes.append((i, j))

        elif self.feed_modes == "one2one":
            print("Creating feed modes, one2one")
            self.feed_modes = []
            for i in range(len(self.rep_modes)):
                self.feed_modes.append((i, i))

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

    def integration_method(self, x_list, embed):
        x_cat = torch.cat(x_list, 1)
        if self.next_mask:
            mask = self.next_mask
            self.next_mask = None
        else:
            mask, self.next_mask = self.get_batch_masks(embed)
        X_batch = torch.mul(mask, x_cat)

        return X_batch

    def forward(self, x_list, embed=False, *args, **kwargs):
        X = self.integration_method(x_list, embed)
        return X

    def __str__(self):
        _str = ""
        _str += str(self.__class__.__name__)
        _str += "\n"
        _str += "Representation modes: {}\n".format(self.unexpanded_rep_modes)
        _str += "Feed modes: {}\n".format(self.feed_modes)
        return _str

class BiWeightedFixed(IntegrationUnitBuilder):
    """
    Sums pointwise or concatenates two vectors, using a weight and it's compliment

    :param integration_mode:    ("sum"|"concat")
                                Integration function, wether the weighted inputs
                                are summed or concatenated
    :param weight_value:        scalar, between 0 and 1
                                Fixed weight value used for the first input (with
                                respect of the order the paths were provided to
                                the dataloader).
                                if None, random value is used
    """


    def __init__(self, integration_mode="sum", weight_value=None, *args, **kwargs):
        super(BiWeightedFixed, self).__init__(*args, **kwargs)
        assert integration_mode in ("sum", "concat"), "Only sum and concat supported"
        if not weight_value:
            weight_value = np.random.random()
        else:
            assert weight_value >= 0, "weight must be greater or equal to 0"
            assert weight_value <= 1, "weight must be less or equal to 1"
        self.weight = weight_value
        self.weight_complement = 1 - self.weight
        self.integration_mode = integration_mode

    def get_weights(self):
        return self.weight

    def integration_function(self, i1, i2):
        if self.integration_mode == "sum":
            return torch.add(i1, i2)
        elif self.integration_mode == "concat":
            return torch.cat([i1, i2], 1)

    def integration_method(self, i1, i2):
        v1_weighted = torch.mul(i1, self.weight)
        v2_weighted = torch.mul(i2, self.weight_complement)
        return self.integration_function(v1_weighted, v2_weighted)

    def forward(self, x_list, *args, **kwargs):
        assert len(x_list) == 2, "BiWeighted integrators only accept two input modalities"
        X = self.integration_method(*x_list)
        return X

    def __str__(self):
        _str = ""
        _str += str(self.__class__.__name__)
        _str += "\n"
        _str += "Integration method: {}\n".format(self.integration_mode)
        _str += "Weight value: {}\n".format(self.weight)
        return _str


class BiWeightedScalarLearnt(BiWeightedFixed):
    """
    Sums pointwise or concatenates two vectors, using a weight and it's compliment.
    Said weight is learnt
    """

    def __init__(self, *args, **kwargs):
        super(BiWeightedScalarLearnt, self).__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.Tensor([self.weight]))
        self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)
        self.start_training()

    def set_headstart_weight(self, headstart_weight):
        self.weight.data[0] = headstart_weight
        self.weight.requires_grad = False
        self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)

    def start_training(self):
        self.weight.requires_grad = True
        self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)

    def integration_method(self, i1, i2):
        self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)
        v1_weighted = torch.mul(i1, self.weight)
        v2_weighted = torch.mul(i2, self.weight_complement)
        return self.integration_function(v1_weighted, v2_weighted)


    def __str__(self):
        _str = ""
        _str += str(self.__class__.__name__)
        _str += "\n"
        _str += "Integration method: {}\n".format(self.integration_mode)
        _str += "Weight value: {}\n".format(self.weight)
        return _str


class BiWeightedDeepLearnt(BiWeightedFixed):
    """
    Sums pointwise or concatenates two vectors, using a weight and it's compliment.
    Said weight is learnt, using a neural network for each of the input vectors,
    summing them and finally puting it through an activation layer

    :param net_params:      List, indicating the network architecture. It must contain
                            only integers, and must have len equal or greater than
                            two. Each integer represents an internal representation
                            dimension, except the first one that represents the
                            input dimension and the last one that represents
                            the output dimension. Every one of this representations
                            will be joined by a fully connected linear layer, so
                            for example the list [280, 1000, 39] means that the
                            input of the network is 280, then a linear layer will
                            take that to 1000 dims, and then a final layer will
                            produce a 39 dimension output vector.

    :param activation_type: ('sigmoid'|'tanh'),
                            activation type of the activation layer that the summed
                            linear projections go through before being returned,
                            default is Sigmoid

    :param init_type:       ('xavier_uni'|'xavier_normal'|'orthogonal'),
                            type of weight initialization, default is 'xavier_uni'
    """

    def __init__(self, net_params, activation_type="sigmoid",
                       init_type='xavier_uni', *args, **kwargs):
        super(BiWeightedDeepLearnt, self).__init__(*args, **kwargs)
        assert activation_type in ('sigmoid', 'tanh')
        assert init_type in ('xavier_uni', 'xavier_normal', 'orthogonal')

        self.input_dim1 = net_params[0][0]
        self.input_dim2 = net_params[1][0]
        self.activation_layer = activation_functions[activation_type]
        self.activation_type = activation_type
        self.init_function = init_functions[init_type]
        self.freezed = False

        self.weight = Variable(torch.rand(1))
        self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)

        self.linear1 = self.build_net(net_params[0], self.activation_layer)
        self.linear2 = self.build_net(net_params[1], self.activation_layer)
        self.apply(self.init_weight_method)
        self.start_training()

    def build_net(self, dimensions_list, activation):
        dimensions_list = expand_dimension_list(dimensions_list)

        layers = []
        for idx in range(len(dimensions_list)-1):
            in_dim = dimensions_list[idx]
            out_dim = dimensions_list[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if idx != len(dimensions_list)-2:
                layers.append(activation) #on the last layer, the activation is
                                          #applied after the sum of both networks

        layers = nn.Sequential(*layers)
        return layers

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = self.init_function
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_type))
            layer.bias.data.fill_(0.0)

    def set_headstart_weight(self, headstart_weight):
        self.weight = Variable(torch.Tensor([headstart_weight]))
        self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)
        self.freezed = True
        for p in self.parameters():
            p.requires_grad = False

    def start_training(self):
        self.freezed = False
        for p in self.parameters():
            p.requires_grad = True

    def compute_attention_weight(self, i1, i2):
        linear1_output = self.linear1(i1)
        linear2_output = self.linear2(i2)
        added = torch.add(linear1_output, linear2_output)
        return self.activation_layer(added)

    def integration_method(self, i1, i2, di1, di2):
        if not self.freezed:
            self.weight = self.compute_attention_weight(di1, di2)
            self.weight_complement = torch.add(torch.mul(self.weight, -1), 1)
        return super(BiWeightedDeepLearnt, self).integration_method(i1, i2)

    def forward(self, x_list, di=None, *args, **kwargs):
        assert len(x_list) == 2, "BiWeighted integrators only accept two input modalities"
        i1 = x_list[0]
        i2 = x_list[1]
        if di:
            assert len(di) == 2, "BiWeighted integrators only accept two asynchronous inputs"
            X = self.integration_method(i1, i2, di[0], di[1])
        else:
            X = self.integration_method(i1, i2, i1, i2)
        return X

    def __str__(self):
        _str = ""
        _str += str(self.__class__.__name__)
        _str += "\n"
        _str += "Integration method: {}\n".format(self.integration_mode)

        if self.input_dim2:
            _str += "Input dims:    ({}, {})\n".format(self.input_dim1,
                                                       self.input_dim2)
        else:
            _str += "Input dims:    ({0}, {0})\n".format(self.input_dim1)
        _str += "Activation:    {}\n".format(self.activation_type)
        _str += "\nLinear 1:\n{}".format(str(self.linear1))
        _str += "\nLinear 2:\n{}".format(str(self.linear2))
        _str += "\nAct Layer:     {}\n".format(str(self.activation_layer))
        return _str


class BiWeightedPreTrained(BiWeightedDeepLearnt):

    def __init__(self, net_1, net_2, net_path1, net_path2,
                       trim_net1_start=None, trim_net1_end=None,
                       trim_net2_start=None, trim_net2_end=None,
                       *args, **kwargs):
        super(BiWeightedPreTrained, self).__init__(*args, **kwargs)

        self.pretrained_1 = self.__load_network(net_1, net_path1, trim_net1_start,
                                                    trim_net1_end, "pre-trained 1")

        self.pretrained_2 = self.__load_network(net_2, net_path2, trim_net2_start,
                                                    trim_net2_end, "pre-trained 2")

        if self.cuda_bool:
            self.pretrained_1.cuda()
            self.pretrained_2.cuda()

    @staticmethod
    def __load_network(network, path, trim_start, trim_end, title = "network"):
        network.load_network(path)
        if trim_start or trim_send:
            network = self.__trim_network(network,
                                        trim_start,
                                        trim_end)
            print("Trimmed {}, new structure:".format(title))
            print(network)
        else:
            network = nn.Sequential(*list(network.children()))
        return network

    @staticmethod
    def __trim_network(network, start_idx, end_idx):
        child = list(network.children())

        if start_idx:
            assert start_idx > 0 and start_idx < end_idx
        else:
            start_idx = 0

        if end_idx:
            assert end_idx > start_idx and end_idx < len(child)
        else:
            end_idx = len(child)

        return nn.Sequential(*child[start_idx:end_idx])


    def integration_method(self, i1, i2, di1, di2):
        di1 = self.pretrained_1(di1)
        di2 = self.pretrained_2(di2)
        return super(BiWeightedPreTrained, self).integration_method(i1, i2, di1, di2)

    def __str__(self):
        _str = super(BiWeightedPreTrained, self).__str__()
        _str += "\nPre-trained 1:\n{}".format(str(self.pretrained_1))
        _str += "\nPre-trained 2:\n{}".format(str(self.pretrained_2))
        return _str
