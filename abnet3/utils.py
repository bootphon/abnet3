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
from collections import defaultdict
import torch.nn as nn

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
        if features[list(features.keys())[0]].dtype == np.float32:
            self.features = features
        else:
            self.features = cast_features(features)

    @staticmethod
    def get_features_between(feature, time, start, end):
        t = np.where(np.logical_and(time >= start,
                                    time <= end))[0]
        return feature[t, :]

    def get(self, f, on, off):
        # check wether filename is string or byte
        filename = f.encode('UTF-8')  # byte
        if filename not in self.times:
            filename = f
        return self.get_features_between(self.features[filename],
                                         self.times[filename], on, off)


def get_dtw_alignment(feat1, feat2):
    distance_array = cosine_distance(feat1, feat2)
    _, _, paths = DTW(feat1, feat2, return_alignment=True,
                      dist_array=distance_array)
    path1, path2 = paths[1:]
    assert len(path1) == len(path2)
    return path1, path2


def read_dataset(dataset_file):
    """
    :param dataset_file: path to the dataset file containing word pairs
    :return: list of the form
     [(file1, start1, end1, f2, s2, e2, pair_type), ...]
    """
    with open(dataset_file, 'r') as fh:
        lines = fh.readlines()
    pairs = []
    for line in lines:
        tokens = line.strip().split(" ")
        assert len(tokens) == 7
        f1, s1, e1, f2, s2, e2, pair_type = tokens
        s1, e1, s2, e2 = float(s1), float(e1), float(s2), float(e2)
        assert pair_type in ['same', 'diff'], \
            'Unsupported pair type {0}'.format(pair_type)
        pairs.append((f1, s1, e1, f2, s2, e2, pair_type))
    return pairs


def group_pairs(pairs):
    """
    Function that groups pairs by pair_type
    :param list pairs: list of pairs [(file1, start1, end1, f2, s2, e2, pair_type), ...]
    :return:
    dictionnary of the form
    {
        'same': [pairs]
        'diff': [pairs]
    }
    """
    grouped_pairs = {'same': [], 'diff': []}
    for f1, s1, e1, f2, s2, e2, pair_type in pairs:
        assert pair_type in grouped_pairs, \
            'Unsupported pair type {0}'.format(pair_type)
        grouped_pairs[pair_type].append((f1, s1, e1, f2, s2, e2))
    return grouped_pairs


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
    return group_pairs(read_dataset(pair_file))


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

def cast_features(features, target_type=np.float32):
    """
    cast features to float32, as this is the currently supported type.
    """
    for item in features:
        features[item] = features[item].astype(target_type)
    print('Casted features to correct type np.float32')
    return features


def read_vad_file(path):
    """
    Read vad file of the form
    https://github.com/bootphon/Zerospeech2015/blob/master/english_vad.txt
    returns a dictionnary of the form {file: [[s1, e1], [s2, e2], ...]}
    """
    with open(path, 'r') as f:
        lines = [line.strip().split(',') for line in f]
        lines = lines[1:]  # skip header
        lines = [(name, float(s), float(e)) for name, s, e in lines]

        dict_vad = defaultdict(list)

        for name, s, e in lines:
            dict_vad[name].append([s, e])

    return dict_vad


def progress(max_number, every=0.1, title=""):
    """
    print progress of a process.
    This function returns another function,
    that has to be called at every iteration, and will print progress.

    # Usage:

    print_progress = progress(100, title="my process")

    for i in range(100):
        do_stuff()
        print_progress(i) # this will print progression every time
                          # we reach 10% more (default)
    """
    next_progress = 0

    def print_progress(current_progress):
        nonlocal next_progress
        current = current_progress / max_number
        if current >= next_progress:
            print("Progress: {:.1f}% of process {}".format(next_progress*100, title))
            next_progress = (current // every) * every + every
    return print_progress

class EmbeddingObserver(object):
    '''
    Observer pattern implementation for the embedding phase. Can be used to
    store and save stages of the model's intern response to the features being
    embedded. It saves the states on the features group with the same items and
    times that were used to save the embeddings embeddings for analysis.
    '''

    def __init__(self):
        self.intern_responses = []

    def register_response(self, response):
        response = response.cpu()
        self.intern_responses.append(response.data.numpy())

    def save(self, path, items, times):
        '''
        Save the internal responses

        :param path:    path to save the internal responses
        :param items:   same items used to save the embeddings
        :param times:   same times used to save embeddings

        '''
        data = h5features.Data(items, times, self.intern_responses, check=True)
        with h5features.Writer(path) as fh:
            fh.write(data, 'features')

class SequentialPartialSave(nn.Sequential):
    '''
    add description
    '''

    def __init__(self, *args, **kwargs):
        super(SequentialPartialSave, self).__init__(*args, **kwargs)
        self.partial_results = self.create_partial_dict()

    def create_partial_dict(self):
        partial_dict = {}
        for i in range(len(self.layers)):
            partial_dict[i] = 0
        return partial_dict

    def get_partial_result(self, index):
        return self.partial_results[index]

    def forward(self, input):
        i = 0
        
        for module in self._modules.values():
            self.partial_results[i] = input
            input = module(input)
            i += 1

        self.partial_results[i] = input
        return input
