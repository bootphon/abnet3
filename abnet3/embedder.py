#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:04:56 2017

@author: Rachid Riad
"""

from abnet3.model import *
import torch
from torch.autograd import Variable
import h5features
from abnet3.utils import read_feats


class EmbedderBuilder:
    """Generic Embedder class for ABnet3
    
    """
    def __init__(self, network, network_path, feature_path=None,
                 output_path=None, cuda=True):
        self.network = network
        self.network_path = network_path
        self.feature_path = feature_path
        self.output_path = output_path
        self.cuda = cuda
        
    def embed(self):
        """ Embed method to embed features based on a saved network
        
        """
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
            emb, _ = self.network(feat_torch,feat_torch).cpu()
            embeddings.append(emb.data.numpy())
        
        data = h5features.Data(items, times, embeddings, check=True)
        with h5features.Writer(self.output_path) as fh:
            fh.write(data, 'features')
        
