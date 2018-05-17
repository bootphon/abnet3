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
from abnet3.utils import SequentialPartialSave, expand_dimension_list, to_ordinal
# from graphviz import Digraph

activation_functions = {'relu': nn.ReLU,
                        'sigmoid': nn.Sigmoid,
                        'tanh': nn.Tanh}

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
                 output_path=None,
                 softmax=False):
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
        self.softmax = softmax
        # Pass forward network functions

        activation = activation_functions[activation_layer]

        # input layer
        input_layer = [
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=p_dropout),
        ]
        if self.batch_norm:
            input_layer.append(nn.BatchNorm1d(hidden_dim))
        input_layer.append(activation())
        self.input_emb = nn.Sequential(*input_layer)

        # hidden layers
        self.hidden_layers = []
        for idx in range(self.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers.append(activation())
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # output layer
        output_layer = [
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=p_dropout)]
        if self.batch_norm:
            output_layer.append(nn.BatchNorm1d(output_dim))
        if softmax:
            output_layer.append(nn.Softmax())
        else:
            output_layer.append(activation())
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
        torch.save(self.state_dict(), self.output_path + str(epoch) + '.pth')

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
        input_layer.append(activation())
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
            self.hidden_layers_shared.append(activation())

        for idx in range(self.num_hidden_layers_spk):
            self.hidden_layers_spk.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_spk.append(
                    nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers_spk.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers_spk.append(activation())

        for idx in range(self.num_hidden_layers_phn):
            self.hidden_layers_phn.append(
                    nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers_phn.append(
                    nn.Dropout(p=p_dropout))
            if self.batch_norm:
                self.hidden_layers_phn.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers_phn.append(activation())

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
        output_layer_spk.append(activation())
        self.output_layer_spk = nn.Sequential(*output_layer_spk)

        # output layer phoneme
        output_layer_phn = [
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=p_dropout),
        ]
        if self.batch_norm:
            output_layer_phn.append(nn.BatchNorm1d(output_dim))
        output_layer_phn.append(activation())
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
    pre_integration_net_params : List of lists
        Every list refers to one of the modality networks. This lists must
        contain only integers. Each integer represents an internal
        representation dimension, except the first one that represents the input
        dimension and the last one that represents the output dimension. Every
        one of this representations will be joined by a fully connected linear
        layer. For contiguous representations of the same dimension, instead of
        writing it multiple times, a tuple can be passed of the form
        (dimension, number of times it appears). If None, the integration unit
        will be the first layer
    post_integration_net_params : List
        List, indicating the dimensions of the after integration network. It
        should have the same form as the pre integration lists
        If None, the integration unit will be the last layer
    p_dropout: Float
        Probability to drop a unit during forward training (common to all nets)
    asynchronous_integration_index: Int
        Only available with integrators which use inputs for attention model.
        This index indicates from which layer those inputs will be taken.
        If None, they will be the same ones being joined.
    batch_norm: Bool
        Add batch normalization layer on the first layer(s)
    type_init: String
        Type of weight initialization (common to all nets)
    activation_layer: String
        Type of activation layer (common to all nets)
    output_path: String
        Path to save network, params
    """

    def __init__(self, integration_unit=None,
                       pre_integration_net_params=None,
                       post_integration_net_params=None,
                       attention_lr=None,
                       asynchronous_integration_index=None,
                       p_dropout=0, batch_norm=False,
                       type_init='xavier_uni', activation_layer=None,
                       output_path=None, *args, **kwargs):
        super(MultimodalSiameseNetwork, self).__init__(*args, **kwargs)
        assert activation_layer in ('relu', 'sigmoid', 'tanh')
        assert type_init in ('xavier_uni', 'xavier_normal', 'orthogonal')
        assert integration_unit is not None, 'If only using one input, use original SiameseNetwork'


        if asynchronous_integration_index:
            assert asynchronous_integration_index >= 0,\
                    '''
                    asynchronous integration index must be greater than 0
                    '''
            assert asynchronous_integration_index < len(pre_integration_net_params[0]) - 1,\
                    '''
                    asynchronous integration index must be less than number of
                    layers on the pre integration network
                    '''

            assert pre_integration_net_params,\
                    '''
                    If asynchronous integration index provided, then there must
                    exist pre integration networks
                    '''


        self.activation_layer = activation_layer
        self.batch_norm = batch_norm
        self.type_init = type_init
        self.p_dropout = p_dropout
        self.output_path = output_path
        self.integration_unit = integration_unit
        self.attention_lr = attention_lr
        self.asynchronous_integration_index = asynchronous_integration_index


        # Pass forward network functions
        activation = activation_functions[activation_layer]

        #Create nets
        if pre_integration_net_params:
            self.pre_nets = []
            for pre_net_params in pre_integration_net_params:

                self.pre_nets.append(SequentialPartialSave(*self.build_net(
                                                            pre_net_params,
                                                            activation)))

            self.pre = True
        else:
            self.pre = False

        if post_integration_net_params:
            self.post_net = nn.Sequential(*self.build_net(post_integration_net_params,
                                                          activation))
            self.post = True
        else:
            self.post = False

        #Init nets
        self.apply(self.init_weight_method)

    def build_net(self, dimensions_list, activation):

        dimensions_list = expand_dimension_list(dimensions_list)
        # layers
        layers = []
        for idx in range(len(dimensions_list)-1):
            in_dim = dimensions_list[idx]
            out_dim = dimensions_list[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Dropout(p=self.p_dropout))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation())

        return layers

    def init_weight_method(self, layer):
        if isinstance(layer, nn.Linear):
            init_func = init_functions[self.type_init]
            init_func(layer.weight.data,
                      gain=nn.init.calculate_gain(self.activation_layer))
            layer.bias.data.fill_(0.0)

    def cuda(self):
        super(MultimodalSiameseNetwork, self).cuda()
        for pre_net in self.pre_nets:
            pre_net.cuda()

    def parameters(self):
        network_params = []
        if self.pre:
            for pre_net in self.pre_nets:
                network_params += list(pre_net.parameters())
        if self.post:
            network_params += list(self.post_net.parameters())

        if self.attention_lr:
            return [{'params': network_params},
                    {'params': self.integration_unit.parameters(),
                                                        'lr': self.attention_lr}
                    ]

        else:
            network_params += list(self.integration_unit.parameters())
            return [{'params': network_params}]

    def freeze_training(self):
        for p in super(MultimodalSiameseNetwork, self).parameters():
            p.requires_grad = False


    def forward_once(self, x_list):
        """Simple forward pass for one instance x_list, which contains multiple
        inputs

        """
        partial_results = x_list
        if self.pre:
            assert len(x_list) == len(self.pre_nets), "Number of inputs: "+\
                                                      "{} doesn't ".format(len(x_list))+\
                                                      "match number of pre_integration "+\
                                                      "nets: {}".format(len(self.pre_nets))
            partial_results = []
            for _input, pre_net in zip(x_list, self.pre_nets):
                partial_results.append(pre_net(_input))

        if self.asynchronous_integration_index is not None:
            attention_inputs = []
            for pre_net in self.pre_nets:
                attention_inputs.append(pre_net.get_partial_result(
                                                self.asynchronous_integration_index
                                                ))

            output = self.integration_unit(partial_results,
                                           diff_input = attention_inputs)
        else:
            output = self.integration_unit(partial_results)

        if self.post:
            output = self.post_net(output)
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
        return {'params': self.__dict__, 'class_name': self.__class__.__name__,
                'architecture': self.architecture_str()}

    def save_network(self, epoch=''):
        torch.save(self.state_dict(), self.output_path + epoch + 'network.pth')
        print("Saved network")
        self.integration_unit.save()
        print("Saved integration unit")

    def load_network(self, path=None):
        self.load_state_dict(torch.load(path+'network.pth'))
        print("Done loading network")
        self.integration_unit.load(path)
        print("Done loading integration unit")

    def architecture_str(self):
        _str = "Multimodal Siamese Architecture"
        if self.pre:
            net_index = 1
            for pre_net in self.pre_nets:
                _str += "\nPre Net {}:\n".format(net_index)
                _str += str(pre_net)
                _str += "\n"
                net_index +=1

        _str += "\nIntegration Unit:\n"
        _str += str(self.integration_unit)
        if self.asynchronous_integration_index is not None:
            _str += "\nAsynchronous integration using "
            if self.asynchronous_integration_index == 0:
                _str += "raw features\n"
            else:
                _str += "{} layer output\n".format(to_ordinal(
                                            self.asynchronous_integration_index))


        if self.post:
            _str += "\nPost Net:\n"
            _str += str(self.post_net)
            _str += "\n"

        return _str




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
