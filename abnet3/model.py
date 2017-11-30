#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is the zoo of models for ABnet3. It is composed of
the different type of architectures that are possible so far.

The networks have a consequent number of parameters that need to be saved to
replicate results.

"""


import torch
import torch.nn as nn
import torch.nn.init as nn.init
from torch.autograd import Variable
import numpy as np
# from graphviz import Digraph

activation_functions = {'relu': nn.ReLU(inplace=True),
                        'sigmoid': nn.Sigmoid(),
                        'tanh': nn.Tanh()}

init_functions = {'xavier_uni': nn.init.xavier_uniform,
                  'xavier_normal': nn.init.xavier_normal,
                  'orthogonal': torch.nn.init.orthogonal}


class NetworkBuilder(nn.Module):
    """Generic Neural Network Model class

    """
    def __init__(self, *args, **kwargs):
        super(NetworkBuilder, self).__init__()

    def forward_once(self, *args, **kwargs):
        raise NotImplementedError('Unimplemented forward_once for class:',
                                  self.__class__.__name__)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Unimplemented forward for class:',
                                  self.__class__.__name__)

    def whoami(self, *args, **kwargs):
        """Output description for the neural network and all parameters

        """
        raise NotImplementedError('Unimplemented whoami for class:',
                                  self.__class__.__name__)

    def save_network(self, *args, **kwargs):
        """Save network weights

        """
        raise NotImplementedError('Unimplemented save_network for class:',
                                  self.__class__.__name__)

    def load_network(self, *args, **kwargs):
        """Load network weights

        """
        raise NotImplementedError('Unimplemented load_network for class:',
                                  self.__class__.__name__)

    def init_weight_method(self, *args, **kwargs):
        """Init network weights

        """
        raise NotImplementedError('Unimplemented init_weight_method' +
                                  'for class:',
                                  self.__class__.__name__)

    def plot_network(self, *args, **kwargs):
        """Vizualize network graphviz

        """
        raise NotImplementedError('Unimplemented plot_network for class:',
                                  self.__class__.__name__)


class SiameseNetwork(NetworkBuilder):
    """Siamese neural network Architecture

    Parameters
    ----------
    input_dim : Int
        Input dimension to the siamese network
    num_hidden_layers: Int
        Number of hidden layers
    hidden_dim: Int, or list of Int
        Dimension of hidden layers
    output_dim: Int
        Dimension of output layer
    p_dropout: Float
        Probability to drop a unit during forward training
    batch_norm: Bool
        Add batch normalization layer
    type_init: String
        Type of weight initialization
    activation_layer: String
        Type of activation layer
    output_path: String
        Path to save network, params
    """
    def __init__(self, input_dim=None, num_hidden_layers=None, hidden_dim=None,
                 output_dim=None, p_dropout=0.1, batch_norm=False,
                 type_init='xavier_uni', activation_layer=None,
                 output_path=None, *args, **kwargs):
        super(SiameseNetwork, self).__init__()
        assert activation_layer in ('relu', 'sigmoid', 'tanh')
        assert type_init in ('xavier_uni', 'xavier_normal', 'orthogonal')
        assert type(input_dim) == int, 'input dim should be int'
        assert type(hidden_dim) == int, 'hidden dim should be int'
        assert type(num_hidden_layers) == int, 'num hidden lay should be int'
        assert type(output_dim) == int, 'output dim should be int'

        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_layer = activation_layer
        self.batch_norm = batch_norm
        self.type_init = type_init
        # Pass forward network functions
        self.input_emb = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(p=p_dropout, inplace=False),
                activation_functions[activation_layer])
        self.hidden_layers = []
        for idx in range(self.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Dropout(p=p_dropout, inplace=False))
            self.hidden_layers.append(activation_functions[activation_layer])

        # * is used for pointing to the list
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(p=p_dropout, inplace=True),
                activation_functions[activation_layer])
        self.output_path = output_path
        self.apply(self.init_weight_method)

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = init_functions[self.type_init]
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_layer))
            layer.bias.data.fill_(0.0)

    def forward_once(self, x):
        """Simple forward pass for one instance x

        """
        output = self.input_emb(x)
        output = self.hidden_layers(output)
        output = self.output_layer(output)
        return output

    def forward(self, input1, input2):
        """Forward pass through the same network

        https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398/2
        reason for design of the siamese
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def whoami(self):
        """Output description for the neural network and all parameters

        """
        return {'params': self.__dict__, 'class_name': self.__class__.__name__}

    def save_network(self, epoch=''):
        torch.save(self.state_dict(), self.output_path + epoch + '.pth')

    def load_network(self, network_path=None):
        self.load_state_dict(torch.load(network_path))


class SiameseMultitaskNetwork(NetworkBuilder):
    """Siamese neural network Architecture for multi-task speaker and
    speech representation

    Parameters
    ----------
    input_dim : Int
        Input dimension to the siamese network
    num_hidden_layers_shared: Int
        Number of hidden layers shared between spk and phn embedding
    num_hidden_layers_spk: Int
        Number of hidden layers shared between spk embedding
    num_hidden_layers_phn: Int
        Number of hidden layers shared between phn embedding
    hidden_dim: Int, or list of Int
        Dimension of hidden layers
    output_dim: Int
        Dimension of output layer for spk embedding and phn embedding
    p_dropout: Float
        Probability to drop a unit during forward training
    batch_norm: Bool
        Add batch normalization layer
    type_init: String
        Type of weight initialization
    activation_layer: String
        Type of activation layer
    output_path: String
        Path to save network, params
    """
    def __init__(self, input_dim=None, num_hidden_layers_shared=None,
                 num_hidden_layers_spk=None,
                 num_hidden_layers_phn=None,
                 hidden_dim=None,
                 output_dim=None, p_dropout=0.1, batch_norm=False,
                 type_init='xavier_uni', activation_layer=None,
                 output_path=None, *args, **kwargs):
        super(SiameseMultitaskNetwork, self).__init__()
        assert activation_layer in ('relu', 'sigmoid', 'tanh')
        assert type_init in ('xavier_uni', 'xavier_normal', 'orthogonal')
        assert type(input_dim) == int, 'input dim should be int'
        assert type(hidden_dim) == int, 'hidden dim should be int'
        assert type(num_hidden_layers_shared) == int
        assert type(num_hidden_layers_spk) == int
        assert type(num_hidden_layers_phn) == int
        assert type(output_dim) == int, 'output dim should be int'

        self.input_dim = input_dim
        self.num_hidden_layers_shared = num_hidden_layers_shared
        self.num_hidden_layers_spk = num_hidden_layers_spk
        self.num_hidden_layers_phn = num_hidden_layers_phn
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_layer = activation_layer
        self.batch_norm = batch_norm
        self.type_init = type_init
        # Pass forward network functions
        self.input_emb = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(p=p_dropout, inplace=False),
                activation_functions[activation_layer])

        self.hidden_layers_shared = []
        self.hidden_layers_spk = []
        self.hidden_layers_phn = []

        for idx in range(self.num_hidden_layers_shared):
            self.hidden_layers_shared.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_shared.append(
                    nn.Dropout(p=p_dropout, inplace=False))
            self.hidden_layers_shared.append(
                    activation_functions[activation_layer])

        for idx in range(self.num_hidden_layers_spk):
            self.hidden_layers_spk.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_spk.append(
                    nn.Dropout(p=p_dropout, inplace=False))
            self.hidden_layers_spk.append(
                    activation_functions[activation_layer])

        for idx in range(self.num_hidden_layers_phn):
            self.hidden_layers_phn.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_phn.append(
                    nn.Dropout(p=p_dropout, inplace=False))
            self.hidden_layers_phn.append(
                    activation_functions[activation_layer])

        # * is used for pointing to the list
        self.hidden_layers_shared = nn.Sequential(*self.hidden_layers_shared)
        self.hidden_layers_spk = nn.Sequential(*self.hidden_layers_spk)
        self.hidden_layers_phn = nn.Sequential(*self.hidden_layers_phn)

        self.output_layer_spk = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(p=p_dropout, inplace=True),
                activation_functions[activation_layer])

        self.output_layer_phn = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(p=p_dropout, inplace=True),
                activation_functions[activation_layer])

        self.output_path = output_path
        self.apply(self.init_weight_method)

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = init_functions[self.type_init]
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_layer))
            layer.bias.data.fill_(0.0)

    def forward_once(self, x):
        """Simple forward pass for one instance x

        """
        output = self.input_emb(x)
        output = self.hidden_layers_shared(output)
        output_spk = self.output_layer_spk(output)
        output_phn = self.output_layer_phn(output)
        return output_spk, output_phn

    def forward(self, input1, input2):
        """Forward pass through the same network

        https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398/2
        reason for design of the siamese
        """
        output_spk1, output_phn1 = self.forward_once(input1)
        output_spk2, output_phn2 = self.forward_once(input2)
        return output_spk1, output_phn1, output_spk2, output_phn2

    def whoami(self):
        """Output description for the neural network and all parameters

        """
        return {'params': self.__dict__, 'class_name': self.__class__.__name__}

    def save_network(self, epoch=''):
        torch.save(self.state_dict(), self.output_path + epoch + '.pth')

    def load_network(self, network_path=None):
        self.load_state_dict(torch.load(network_path))


if __name__ == '__main__':
    sia = SiameseNetwork(input_dim=3, num_hidden_layers=2, hidden_dim=10,
                         output_dim=19, dropout=0.1,
                         activation_layer='relu',
                         batch_norm=True)
    siaMulti = SiameseMultitaskNetwork(input_dim=3, num_hidden_layers_shared=2,
                                       num_hidden_layers_phn=1,
                                       num_hidden_layers_spk=1,
                                       hidden_dim=10,
                                       output_dim=19, dropout=0.1,
                                       activation_layer='relu',
                                       batch_norm=True)
    sia.apply(sia.init_weight_method)
    N_batch = 64
    x1 = Variable(torch.randn(N_batch, 1, 1, 3))
    x2 = Variable(torch.randn(N_batch, 1, 1, 3))
    output1, output2 = sia(x1, x2)
    y = Variable(torch.LongTensor(np.random.choice([1, -1], N_batch)))
    output_spk1, output_phn1, output_spk2, output_phn2, = siaMulti(x1, x2)
#    pl = sia.plot_network()
