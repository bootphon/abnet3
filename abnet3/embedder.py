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


class EmebedderBuilder:
    """Generic Embedder class for ABnet3
    
    """
    def __init__(self, network, feature_path=None, output_path=None):
        self.network = network
        self.feature_path = feature_path
        self.output_path = output_path
    
    def embed(self):
        """ Embed method to embed features based on a save network
        
        """

        with h5features.Reader(self.feature_path, 'features') as fh:
            features = fh.read()

        items = features.items()
        times = features.labels()
        feats = features.features()

        embeddings = []
        for feat in feats:
            feat_torch = Variable(torch.from_numpy(feat))
            emb = self.network.forward_once(feat_torch)
            embeddings.append(emb.data.numpy())
        
        data = h5features.Data(items, times, embeddings, check=True)
        with h5features.Writer(self.output_path) as fh:
            fh.write(data, 'features')
        
