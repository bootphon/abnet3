#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  3 19:22:41 2017

@author: Rachid Riad
"""

import numpy as np
import os
import h5features

def get_var_name(**variable):
    return list(variable.keys())[0]


def normalize_distribution(p):
    """Normalize distribution p for a dictionnary k/v class
    
    """
    assert type(p)==dict, 'Distribution p is not a dictionnary'
    sum_norm = 0.0
    keys = p.keys()
    for key in keys:
        sum_norm += p[key]
    
    for key in keys:
        p[key] = p[key]/sum_norm

    return p
    

# Inspiration from numpy code source 

def cumulative_distribution(distribution):
    """Cumulative sums for multinomial sampling
    
    """
    assert type(distribution) == list, 'distribution variable needs to be list'
    assert np.sum(distribution) == 1.0, 'distribution needs to be normalized'
    
    cdf = np.cumsum(np.array(distribution))
    cdf /= cdf[-1]
    return cdf


def sample_searchidx(cdf, num_samples):
    """Sample indexes based on cdf distribution
    
    """
    uniform_samples = np.random.random_sample(num_samples)
    idx = cdf.searchsorted(uniform_samples, side='right')
    return idx

def print_token(tok):
    """Pretty print token for batches
    
    """
    return "{0} {1:.2f} {2:.2f}".format(tok[0], tok[1], tok[2])


def Parse_Dataset(path):
    """Parse folder for batch names
    
    """
    batches = []
    batches += ([os.path.join(path, add) for add in os.listdir(path) if add.endswith(('.batch'))])
    return batches


class Features_Accessor(object):
    
    def __init__(self, times, features):
        self.times = times
        self.features = features


    def get(self, f, on, off):
        t = np.where(np.logical_and(self.times[f] >= on,
                                    self.times[f] <= off))[0]
        return self.features[f][t, :]



def read_pairs(pair_file):
    with open(pair_file, 'r') as fh:
        lines = fh.readlines()
    pairs = {'same' : [], 'diff' : []}
    for line in lines:
        tokens = line.strip().split(" ")
        assert len(tokens) == 7
        f1, s1, e1, f2, s2, e2, pair_type = tokens
        s1, e1, s2, e2 = float(s1), float(e1), float(s2), float(e2)
        assert pair_type in pairs, \
               'Unsupported pair type {0}'.format(pair_type)
        pairs[pair_type].append((f1, s1, e1, f2, s2, e2))
    return pairs


def read_feats(features_file, align_features_file=None):
    with h5features.Reader(features_file, 'features') as fh:
        features = fh.read()  # load all at once here...
    times = features.dict_labels()
    feats = features.dict_features()
    feat_dim = feats[list(feats.keys())[0]].shape[1]
    features = Features_Accessor(times, feats)
    if align_features_file is None:
        align_features = None
    else:
        with h5features.Reader(features_file, 'features') as fh:
            align_features = fh.read()  # load all at once here...
        times = align_features.dict_labels()
        feats = align_features.dict_features()
        align_features = Features_Accessor(times, feats)
    return features, align_features, feat_dim







