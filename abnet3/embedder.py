#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is composed of the different modules for embedding features
based on neural network.

"""

import numpy as np
import torch
from torch.autograd import Variable
import h5features
import argparse

from abnet3.utils import read_feats, EmbeddingObserver
from abnet3.model import *


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
    def __init__(self, network=None, network_path=None, feature_path=None,
                 output_path=None, cuda=True):
        if network is None:
            raise ValueError("network is None.")
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

    def __init__(self, *args, **kwargs):
        super(EmbedderSiamese, self).__init__(*args, **kwargs)

    def embed(self):
        """ Embed method to embed features based on a saved network

        """
        if self.network_path is not None:
            self.network.load_network(self.network_path)
        self.network.eval()
        print("Done loading network weights")

        with h5features.Reader(self.feature_path, 'features') as fh:
            features = fh.read()

        items = features.items()
        times = features.labels()
        feats = features.features()
        print("Done loading input feature file")

        embeddings = []
        for feat in feats:
            if feat.dtype != np.float32:
                feat = feat.astype(np.float32)
            feat_torch = Variable(torch.from_numpy(feat), volatile=True)
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

    def __init__(self, *args, **kwargs):
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
            if feat.dtype != np.float32:
                feat = feat.astype(np.float32)
            feat_torch = Variable(torch.from_numpy(feat), volatile=True)
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

class MultimodalEmbedder(EmbedderBuilder):
    """
    Embedder class for multimodal siamese network
    """

    def __init__(self, *args, **kwargs):
        super(MultimodalEmbedder, self).__init__(*args, **kwargs)
        self.observers = [] #tuples list, of the form (EmbedderObserver,
                                                      #function to get the data,
                                                      #path to be saved)

        if type(self.network.integration_unit) == abnet3.integration.BiWeightedLearnt:
            print("Placing observer to save learnt attention weights")
            self.observers.append((EmbeddingObserver(),
                                   self.network.integration_unit.get_weights,
                                   self.output_path + "attention_weights.pth"))

    def embed(self):
        """
        Embed method to embed features based on a saved network
        """

        if self.network_path is not None:
            self.network.load_network(self.network_path)
        self.network.eval()

        if self.cuda:
            self.network.cuda()

        items = None
        times = None
        features_list = []
        for path in self.feature_path:
            with h5features.Reader(path, 'features') as fh:
                features = fh.read()
                features_list.append(features.features())
                check_items = features.items()
                check_times = features.labels()
            if not items:
                items = check_items
                #TODO: assert items == check_items, "Items inconsistency found on path {}".format(path)
            if not times:
                times = check_times
                #TODO: assert times == check_times, "Times inconsistency found on path {}".format(path)

        print("Done loading input feature file")

        zipped_feats = zip(*features_list)
        embeddings = []
        for feats in zipped_feats:
            modes_list = []
            for feat in feats:
                if feat.dtype != np.float32:
                    feat = feat.astype(np.float32)
                feat_torch = Variable(torch.from_numpy(feat), volatile=True)
                if self.cuda:
                    feat_torch = feat_torch.cuda()
                modes_list.append(feat_torch)
            emb, _ = self.network(modes_list, modes_list)
            emb = emb.cpu()
            embeddings.append(emb.data.numpy())

            for observer_tuple in self.observers:
                observer_tuple[0].register_response(observer_tuple[1]())

        data = h5features.Data(items, times, embeddings, check=True)
        with h5features.Writer(self.output_path) as fh:
            fh.write(data, 'features')

        for observer_tuple in self.observers:
            observer_tuple[0].save(observer_tuple[2], items, times)
