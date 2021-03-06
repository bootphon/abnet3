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
from abnet3.integration import BiWeightedDeepLearnt


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
    batch_size: for embedding, to avoid going over gpu memory limit

    """
    def __init__(self, network=None, network_path=None, feature_path=None,
                 output_path=None, cuda=True, batch_size=5000):
        if network is None:
            raise ValueError("network is None.")
        self.network = network
        self.network_path = network_path
        self.feature_path = feature_path
        self.output_path = output_path
        self.cuda = cuda
        self.batch_size = batch_size

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

        if self.cuda:
            self.network.cuda()
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
            n_batches = len(feat) // self.batch_size + 1
            batches_feat = np.array_split(feat, n_batches)
            outputs = []
            for b_feat in batches_feat:
                feat_torch = Variable(torch.from_numpy(b_feat), volatile=True)
                if self.cuda:
                    feat_torch = feat_torch.cuda()
                emb, _ = self.network(feat_torch, feat_torch)
                emb = emb.cpu()
                outputs.append(emb.data.numpy())
            outputs = np.vstack(outputs)
            embeddings.append(outputs)

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

        if self.cuda:
            self.network.cuda()

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

        if isinstance(self.network.integration_unit, BiWeightedDeepLearnt):
            print("Placing observer to save learnt attention weights")
            self.observers.append(EmbeddingObserver(
                                 self.network.integration_unit.get_weights,
                                 self.output_path+"attention_weights.features"))

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
            if not times:
                times = check_times

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

            #Register activity on observer
            for observer in self.observers:
                observer.register_status()

        data = h5features.Data(items, times, embeddings, check=True)
        with h5features.Writer(self.output_path + "embedded.features") as fh:
            fh.write(data, 'features')

        #Save observer registers
        for observer in self.observers:
            observer.save(items, times)
