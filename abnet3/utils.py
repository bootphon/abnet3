#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script is composed of the different utilities functions for the
different parts for the ABnet3 package.

TODO: list the different utilis functions

"""

import numpy as np
import os
import h5features
from dtw import DTW
import scipy


def get_var_name(**variable):
    return list(variable.keys())[0]


def read_spkid_file(spkid_file):
    with open(spkid_file, 'r') as fh:
        lines = fh.readlines()
    spk = {}
    for line in lines:
        fid, spkid = line.strip().split(" ")
        assert not(fid in spk)
        spk[fid] = spkid
    return spk


def read_spk_list(spk_file):
    with open(spk_file, 'r') as fh:
        lines = fh.readlines()
    return [line.strip() for line in lines]


def cosine_distance(x, y):
    assert (x.dtype == np.float64 and y.dtype == np.float64) or (
        x.dtype == np.float32 and y.dtype == np.float32)
    x2 = np.sqrt(np.sum(x ** 2, axis=1))
    y2 = np.sqrt(np.sum(y ** 2, axis=1))
    ix = x2 == 0.
    iy = y2 == 0.
    d = np.dot(x, y.T) / (np.outer(x2, y2))
    # DPX: to prevent the stupid scipy to collapse the array into scalar
    if d.shape == (1, 1):
        d = np.array([[np.float64(scipy.arccos(d) / np.pi)]])
    else:
        # costly in time (half of the time), so check if really useful for dtw
        d = np.float64(scipy.arccos(d) / np.pi)

    d[ix, :] = 1.
    d[:, iy] = 1.
    for i in np.where(ix)[0]:
        d[i, iy] = 0.
    assert np.all(d >= 0)
    return d


def normalize_distribution(p):
    """Normalize distribution p for a dictionnary k/v class

    """
    assert type(p) == dict, 'Distribution p is not a dictionnary'
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
    assert type(distribution) == dict, 'distribution variable needs to be dict'
    values = list(distribution.values())
    cdf = np.cumsum(np.array(values))
    cdf /= cdf[-1]
    return cdf


def sample_searchidx(cdf, num_samples):
    """Sample indexes based on cdf distribution

    """
    uniform_samples = np.random.random_sample(int(num_samples))
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
    batches += ([os.path.join(path, add) for add in os.listdir(path)
                if add.endswith(('.batch'))])
    return batches


class Features_Accessor(object):

    def __init__(self, times, features):
        self.times = times
        self.features = features

    def get(self, f, on, off):
        t = np.where(np.logical_and(self.times[f.encode('UTF-8')] >= on,
                                    self.times[f.encode('UTF-8')] <= off))[0]
        return self.features[f.encode('UTF-8')][t, :]


def get_dtw_alignment(feat1, feat2):
    distance_array = cosine_distance(feat1, feat2)
    _, _, paths = DTW(feat1, feat2, return_alignment=True,
                      dist_array=distance_array)
    path1, path2 = paths[1:]
    assert len(path1) == len(path2)
    return path1, path2


def read_pairs(pair_file):
    """

    :param pair_file: path to the batch file containing word pairs
    :return: dictionnary of the form
    {
        'same': [pairs]
        'diff': [pairs]
    }

    where a pair is a tuple (file1, start1, end1, file2, start2, end2)

    """
    with open(pair_file, 'r') as fh:
        lines = fh.readlines()
    pairs = {'same': [], 'diff': []}
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
