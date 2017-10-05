#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:24:59 2017

@author: Rachid Riad
"""

from abnet3.model import *
from abnet3.loss import *
from abnet3.sampler import *
from abnet3.utils import *
import numpy as np
import h5features

class TrainerBuilder(object):
    """Generic Trainer class for ABnet3
    
    """
    def __init__(self, sampler, network, loss, feature_path=None,
                 num_epochs=200, patience=20):
        super(TrainerBuilder, self).__init__()
        self.sampler = sampler
        self.network = network
        self.loss = loss
        self.feature_path = feature_path
        self.num_epochs = num_epochs
        self.patience = patience
        
    def whoami(self):
        return {'params':self.__dict__,
                'network':self.network.whoami(),
                'loss':self.loss.whoami(),
                'sampler':self.sampler.whoami(),
                'class_name': self.__class__.__name__}
    
    def train(self):
        """Train function 
    
        """
        raise NotImplementedError('Unimplemented train for class:',
                          self.__class__.__name__)

       
class TrainerSiamese(TrainerBuilder):
    """Siamese Trainer class for ABnet3
    
    """
    def __init__(self):
        super(TrainerSiamese, self).__init__() 
        assert type(self.sampler) == 'abnet3.sampler.SamplerClusterSiamese'
        assert type(self.network) == 'abnet3.model.SiameseNetwork'
    
    def prepare_batch_from_pair_words(self, pairs_path, features, seed=0):
        """Prepare a batch in Pytorch format based on a batch file
        
        """
        #TODO : Encapsulate X preparation in another function 
        #TODO : Replace Numpy operation by Pytorch operation
        pairs = read_pairs(pairs_path)
        token_feats = {}
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            token_feats[f1, s1, e1] = True
            token_feats[f2, s2, e2] = True
        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            token_feats[f1, s1, e1] = True
            token_feats[f2, s2, e2] = True
        # 2. fill in features
        for f, s, e in token_feats:
            token_feats[f, s, e] = features.get(f, s, e)
        # 3. align features for each pair
        X1, X2, y = [], [], []
        ## get features for each same pair based on DTW alignment paths
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            if (s1>e1) or (s2>e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            try:
                path1, path2 = get_dtw_alignment(feat1, feat2)
            except:
                continue

            X1.append(feat1[path1,:])
            X2.append(feat2[path2,:])
            y.append(np.ones(len(path1)))
            
        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            if (s1>e1) or (s2>e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            n1 = feat1.shape[0]
            n2 = feat2.shape[0]
            X1.append(feat1[:min(n1, n2),:])
            X2.append(feat2[:min(n1, n2),:])
            y.append(-1*np.ones(min(n1, n2)))
        
        ## concatenate all features
        X1, X2, y = np.vstack(X1), np.vstack(X2), np.concatenate(y)
        np.random.seed(seed)
        n_pairs = len(y)
        ind = np.random.permutation(n_pairs)
        y = Variable(torch.from_numpy(y[ind]))
        X1 = Variable(torch.from_numpy(X1[ind,:]))
        X2 = Variable(torch.from_numpy(X2[ind,:]))
        return X1, X2, y
    
    
    def train(self):
        return 0


if __name__ == '__main__':
    
    sia = SiameseNetwork(input_dim=3,num_hidden_layers=2,hidden_dim=10,
                     output_dim=19,dropout=0.1,
                     activation_function=nn.ReLU(inplace=True),
                     batch_norm=True)
    sam = SamplerClusterSiamese()
    loss = coscos2()
    tra = TrainerBuilder(sam,sia,loss)
    
    


        
        
        