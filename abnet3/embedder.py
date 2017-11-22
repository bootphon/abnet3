#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is composed of the different modules for embedding features
based on neural network.

It will generate text files in two folders 'train_pairs' and 'dev_pairs' in the
correct format for the Neural Network.

"""

from abnet3.model import *
import torch
from torch.autograd import Variable
import h5features
from abnet3.utils import read_feats


class EmbedderBuilder:
    """Generic Embedder class for ABnet3

    Parameters
    ----------
    network : ABnet3 network
        ABnet3 network trained or not, to produce the embedding
    network_path : string
        Path to saved network
    feature_path : string
        Path to features in h5f format
    output_path: string
        Output path
    cuda: Bool
        If Gpu and Cuda are available

    """
    def __init__(self, network, network_path=None, feature_path=None,
                 output_path=None, cuda=True):
        self.network = network
        self.network_path = network_path
        self.feature_path = feature_path
        self.output_path = output_path
        self.cuda = cuda

    def embed(self):
        raise NotImplementedError('Unimplemented embed for class:',
                                  self.__class__.__name__)


class EmbedderSiamese(EmbedderBuilder):
    """Embedder class for siamese network on monotask

    """

    def __init__(self, avg=True, *args, **kwargs):
        super(EmbedderSiamese, self).__init__(*args, **kwargs)

    def embed(self):
        """ Embed method to embed features based on a saved network

        """
        if self.network_path is not None:
            self.network.load_network(self.network_path)
        self.network.eval()

        with h5features.Reader(self.feature_path, 'features') as fh:
            features = fh.read()

        items = features.items()
        times = features.labels()
        feats = features.features()

        embeddings = []
        for feat in feats:
            feat_torch = Variable(torch.from_numpy(feat))
            if self.cuda:
                feat_torch = feat_torch.cuda()
            emb, _ = self.network(feat_torch, feat_torch)
            emb = emb.cpu()
            embeddings.append(emb.data.numpy())

        data = h5features.Data(items, times, embeddings, check=True)
        with h5features.Writer(self.output_path) as fh:
            fh.write(data, 'features')


class EmbedderSiameseMultitask(EmbedderBuilder):
    """Embedder class for siamese network on multitask

    """

    def __init__(self, avg=True, *args, **kwargs):
        super(EmbedderSiameseMultitask, self).__init__(*args, **kwargs)

    def embed(self):
        """ Embed method to embed features based on a saved network

        """
        if self.network_path is not None:
            self.network.load_network(self.network_path)
        self.network.eval()

        with h5features.Reader(self.feature_path, 'features') as fh:
            features = fh.read()

        items = features.items()
        times = features.labels()
        feats = features.features()

        embeddings_spk, embeddings_phn = [], []
        for feat in feats:
            feat_torch = Variable(torch.from_numpy(feat))
            if self.cuda:
                feat_torch = feat_torch.cuda()
            emb_spk, emb_phn, _, _ = self.network(feat_torch, feat_torch)
            emb_spk = emb_spk.cpu()
            emb_phn = emb_phn.cpu()
            embeddings_spk.append(emb_spk.data.numpy())
            embeddings_phn.append(emb_phn.data.numpy())

        data_spk = h5features.Data(items, times, embeddings_spk, check=True)
        data_phn = h5features.Data(items, times, embeddings_phn, check=True)

        with h5features.Writer(self.output_path+'.spk') as fh:
            fh.write(data_spk, 'features')

        with h5features.Writer(self.output_path+'.phn') as fh:
            fh.write(data_phn, 'features')
