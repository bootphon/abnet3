#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is the zoo of models for ABnet3. It is composed of
the different type of architectures that are possible so far.

The networks have a consequent number of parameters that need to be saved to
replicate results.

"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# from graphviz import Digraph

activation_functions = {'relu': nn.ReLU(),
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
                 output_path=None):
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

        activation = activation_functions[activation_layer]

        # input layer
        input_layer = [
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=p_dropout),
        ]
        if self.batch_norm:
            input_layer.append(nn.BatchNorm1d(hidden_dim))
        input_layer.append(activation)
        self.input_emb = nn.Sequential(*input_layer)

        # hidden layers
        self.hidden_layers = []
        for idx in range(self.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers.append(activation)
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # output layer
        output_layer = [
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=p_dropout)]
        if self.batch_norm:
            output_layer.append(nn.BatchNorm1d(output_dim))
        output_layer.append(activation)
        self.output_layer = nn.Sequential(*output_layer)
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
                 output_path=None):
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
        activation = activation_functions[activation_layer]

        # input layer
        input_layer = [
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=p_dropout),
        ]
        if self.batch_norm:
            input_layer.append(nn.BatchNorm1d(hidden_dim))
        input_layer.append(activation)
        self.input_emb = nn.Sequential(*input_layer)

        self.hidden_layers_shared = []
        self.hidden_layers_spk = []
        self.hidden_layers_phn = []

        for idx in range(self.num_hidden_layers_shared):
            self.hidden_layers_shared.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_shared.append(
                    nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers_shared.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers_shared.append(
                    activation_functions[activation_layer])

        for idx in range(self.num_hidden_layers_spk):
            self.hidden_layers_spk.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_spk.append(
                    nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers_spk.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers_spk.append(
                    activation_functions[activation_layer])

        for idx in range(self.num_hidden_layers_phn):
            self.hidden_layers_phn.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_phn.append(
                    nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers_phn.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers_phn.append(
                    activation_functions[activation_layer])

        # * is used for pointing to the list
        self.hidden_layers_shared = nn.Sequential(*self.hidden_layers_shared)
        self.hidden_layers_spk = nn.Sequential(*self.hidden_layers_spk)
        self.hidden_layers_phn = nn.Sequential(*self.hidden_layers_phn)

        # output layer speaker
        output_layer_spk = [
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=p_dropout),
        ]
        if self.batch_norm:
            output_layer_spk.append(nn.BatchNorm1d(output_dim))
        output_layer_spk.append(activation)
        self.output_layer_spk = nn.Sequential(*output_layer_spk)

        # output layer phoneme
        output_layer_phn = [
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=p_dropout),
        ]
        if self.batch_norm:
            output_layer_phn.append(nn.BatchNorm1d(output_dim))
        output_layer_phn.append(activation)
        self.output_layer_phn = nn.Sequential(*output_layer_phn)

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


class MultimodalSiameseNetwork(NetworkBuilder):
    """Multimodal Siamese neural network Architecture

    Parameters
    ----------
    integration_unit: Integration Class
        Integration unit, which joins the inputs
    pre_integration_net_params : List of tuples
        Every tuple refers to one of the modality networks, previous to integration
        it should have this form:
            tuple[0] : Int, input dim of said network
            tuple[1] : Int, number of hidden layers of said network
            tuple[2] : Int, hidden layers dim of said network
            tuple[3] : Int, output dim of said network
        If None, the integration unit will be the first layer
    post_integration_net_params : Tuple
        Tuple, indicating the parameters of the after integration network. It
        should have the same form as the pre integration tuples
        If None, the integration unit will be the last layer
    p_dropout: Float
        Probability to drop a unit during forward training (common to all nets)
    batch_norm: Bool
        Add batch normalization layer on the first layer(s)
    type_init: String
        Type of weight initialization (common to all nets)
    activation_layer: String
        Type of activation layer (common to all nets)
    output_path: String
        Path to save network, params
    """
    #TODO: add possibility of different activation layers, different dropouts,
    #      different activations, etc...
    #TODO: support more than 2 inputs

    def __init__(self, integration_unit,
                       pre_integration_net_params=None,
                       post_integration_net_params=None,
                       p_dropout=0, batch_norm=False,
                       type_init='xavier_uni', activation_layer=None,
                       output_path=None, *args, **kwargs):
        super(MultimodalSiameseNetwork, self).__init__(*args, **kwargs)
        assert activation_layer in ('relu', 'sigmoid', 'tanh')
        assert type_init in ('xavier_uni', 'xavier_normal', 'orthogonal')
        assert not pre_integration_net_params or \
                len(pre_integration_net_params) == 2, 'Only 2 inputs supported for now'


        self.activation_layer = activation_layer
        self.batch_norm = batch_norm
        self.type_init = type_init
        self.p_dropout = p_dropout
        self.output_path = output_path
        self.integration_unit = integration_unit
        # Pass forward network functions

        activation = activation_functions[activation_layer]

        #Create nets

        if pre_integration_net_params:
            self.pre_net1_in, self.pre_net1_hidden, self.pre_net1_out = \
                        self.build_net(*pre_integration_net_params[0], activation)

            self.pre_net2_in, self.pre_net2_hidden, self.pre_net2_out = \
                        self.build_net(*pre_integration_net_params[1], activation)

            self.pre = True
            self.pre_net1 = [self.pre_net1_in, self.pre_net1_hidden, self.pre_net1_out]
            self.pre_net2 = [self.pre_net2_in, self.pre_net2_hidden, self.pre_net2_out]
        else:
            self.pre = False



        if post_integration_net_params:
            self.post_net_in, self.post_net_hidden, self.post_net_out = \
                            self.build_net(*post_integration_net_params, activation)
            self.post = True
            self.post_net = [self.post_net_in, self.post_net_hidden, self.post_net_out]
        else:
            self.post = False

        #Init nets
        self.apply(self.init_weight_method)

    def build_net(self, input_dim, n_hidden, hidden_dim, output_dim, activation):

        # input layer
        input_layer = [
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=self.p_dropout),
        ]
        if self.batch_norm:
            input_layer.append(nn.BatchNorm1d(hidden_dim))
        input_layer.append(activation)
        input_layer = nn.Sequential(*input_layer)

        # hidden layers
        hidden_layers = []
        for idx in range(n_hidden):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.Dropout(p=self.p_dropout))
            if self.batch_norm:
                hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            hidden_layers.append(activation)
        hidden_layers = nn.Sequential(*hidden_layers)

        # output layer
        output_layer = [
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=self.p_dropout)]
        if self.batch_norm:
            output_layer.append(nn.BatchNorm1d(output_dim))
        output_layer.append(activation)
        output_layer = nn.Sequential(*output_layer)
        return input_layer, hidden_layers, output_layer

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = init_functions[self.type_init]
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_layer))
            layer.bias.data.fill_(0.0)

    def net_forward(self, x, input_layer, hidden_layer, output_layer):
        output = input_layer(x)
        output = hidden_layer(x)
        output = output_layer(x)
        return output


    def forward_once(self, x_list):
        """Simple forward pass for one instance x_list, which contains multiple
        inputs

        """
        if self.pre:
            x1 = self.net_forward(x_list[0], *self.pre_net1)
            x2 = self.net_forward(x_list[1], *self.pre_net2)
        output = self.integration_unit([x1, x2])
        if self.post:
            output = self.net_forward(output, *self.post_net)
        return output

    def forward(self, input1, input2):
        """Forward pass through the same network

        https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398/2
        reason for design of the siamese
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def save_network(self, epoch=''):
        torch.save(self.state_dict(), self.output_path + epoch + 'network.pth')
        print("Saved network")
        self.integration_unit.save()
        print("Saved integration unit")

    def load_network(self, path=None):
        self.load_state_dict(torch.load(+'network.pth'))
        print("Done loading network")
        self.integration_unit.load(path)
        print("Done loading integration unit")



if __name__ == '__main__':
    sia = SiameseNetwork(input_dim=3, num_hidden_layers=2, hidden_dim=10,
                         output_dim=19, p_dropout=0.1,
                         activation_layer='relu',
                         batch_norm=True)
    siaMulti = SiameseMultitaskNetwork(input_dim=3, num_hidden_layers_shared=2,
                                       num_hidden_layers_phn=1,
                                       num_hidden_layers_spk=1,
                                       hidden_dim=10,
                                       output_dim=19, p_dropout=0.1,
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
