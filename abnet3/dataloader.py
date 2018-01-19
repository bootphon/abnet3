import numpy as np
import torch
from torch.autograd import Variable
import os
from collections import defaultdict

from abnet3.utils import get_dtw_alignment, \
    Parse_Dataset, read_pairs, read_feats, read_spkid_file

"""
This file contains several dataloaders.

DataLoaderFromBatches: Original method to load data.
It loads the batches of tokens created by the sampler, create the frame pairs, and shuffles inside the batches.

FramesDataLoader: Reads a dataset composed of one file, loads all the frames in memory, and shuffles across the
whole epoch. It then creates batches.

MultitaskDataLoader : default loader for multitask network

"""


class DataLoader:

    def batch_iterator(self, train_mode=True):
        """
        This function returns an iterator over all the batches of data
        :returns iterator
        """
        raise NotImplemented("You must implement batch iterator in DataLoader class.")

    def whoami(self):
        raise NotImplemented("You must implement whoami in DataLoader class")

class DataLoaderFromBatches(DataLoader):
    """
    Original method to load data.
    It loads the batches of tokens created by the sampler,
    create the frame pairs, and shuffles inside the batches.

    """

    def __init__(self, pairs_path, features_path, num_max_minibatches=1000, seed=None):
        """

        :param string pairs_path: path to dataset where the dev_pairs and train_pairs folders are
        :param features_path: path to feature file
        :param int num_max_minibatches: number of batches in each epoch
        :param int seed: for randomness
        """
        self.pairs_path = pairs_path
        self.features_path = features_path
        self.statistics_training = defaultdict(int) # dict with default value 0
        self.seed = seed
        self.num_max_minibatches = num_max_minibatches
        self.features = None

    def __getstate__(self):
        """used for pickle"""

        return (self.pairs_path,
                self.features_path,
                self.statistics_training,
                self.seed,
                self.num_max_minibatches)

    def __setstate__(self, state):
        """used for pickle"""
        self.pairs_path, \
        self.features_path, \
        self.statistics_training, \
        self.seed, \
        self.num_max_minibatches = state

        self.load_features()

    def whoami(self):
        return {
            'params': self.__getstate__(),
            'class_name': self.__class__.__name__
        }

    def load_features(self):
        """
        Load only once the features
        """
        if self.features is None:
            features, align_features, feat_dim = read_feats(self.features_path)
            self.features = features

    def load_frames_from_pairs(self, pairs, seed=0, fid2spk=None):
        """Prepare a batch in Pytorch format based on a batch file
        :param pairs: list of pairs under the form {'same': [pairs], 'diff': [pairs] }
        :param seed: randomness
        :param fid2spk:
            if None, will return X1, X2, y_phones
            If it is the spkid mapping, will return X1, X2, y_phones, y_speaker
        """

        # f are filenames, s are start times, e are end times
        token_feats = {}
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            token_feats[f1, s1, e1] = self.features.get(f1, s1, e1)
            token_feats[f2, s2, e2] = self.features.get(f2, s2, e2)
        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            token_feats[f1, s1, e1] = self.features.get(f1, s1, e1)
            token_feats[f2, s2, e2] = self.features.get(f2, s2, e2)

        # 2. align features for each pair
        X1, X2, y_phn, y_spk = [], [], [], []
        # get features for each same pair based on DTW alignment paths
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            if (s1 > e1) or (s2 > e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            try:
                path1, path2 = get_dtw_alignment(feat1, feat2)
            except e:
                continue

            self.statistics_training['SameType'] += 1

            if fid2spk:
                spk1, spk2 = fid2spk[f1], fid2spk[f2]
                if spk1 is spk2:
                    y_spk.append(np.ones(len(path1)))
                    self.statistics_training['SameTypeSameSpk'] += 1
                else:
                    y_spk.append(-1 * np.ones(len(path1)))
                    self.statistics_training['SameTypeDiffSpk'] += 1

            X1.append(feat1[path1, :])
            X2.append(feat2[path2, :])
            y_phn.append(np.ones(len(path1)))

        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            if (s1 > e1) or (s2 > e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            n1 = feat1.shape[0]
            n2 = feat2.shape[0]
            X1.append(feat1[:min(n1, n2), :])
            X2.append(feat2[:min(n1, n2), :])
            y_phn.append(-1 * np.ones(min(n1, n2)))

            self.statistics_training['DiffType'] += 1

            if fid2spk:
                spk1, spk2 = fid2spk[f1], fid2spk[f2]
                if spk1 is spk2:
                    y_spk.append(np.ones(min(n1, n2)))
                    self.statistics_training['DiffTypeSameSpk'] += 1
                else:
                    y_spk.append(-1 * np.ones(min(n1, n2)))
                    self.statistics_training['DiffTypeDiffSpk'] += 1

        if fid2spk:
            assert len(y_phn) == len(y_spk), 'not same number of labels...'

        # concatenate all features
        X1, X2, y_phn = np.vstack(X1), np.vstack(X2),  np.concatenate(y_phn)
        np.random.seed(seed)
        n_pairs = len(y_phn)

        ind = np.random.permutation(n_pairs)
        X1 = X1[ind, :]
        X2 = X2[ind, :]
        y_phn = y_phn[ind]

        if fid2spk:
            y_spk = np.concatenate(y_spk)[ind]
            return X1, X2, y_spk, y_phn

        return X1, X2, y_phn

    def batch_iterator(self, train_mode=True):
        """Build iteratior next batch from folder for a specific epoch
        This function can be used when the batches were already created
        by the sampler.

        If you use the sampler that didn't create batches, use the
        new_get_batches function
        Returns batches of the form (X1, X2, y)

        """
        if train_mode:
            batch_dir = os.path.join(self.pairs_path,
                                     'train_pairs')
        else:
            batch_dir = os.path.join(self.pairs_path,
                                     'dev_pairs')
        # load features
        self.load_features()

        batches = Parse_Dataset(batch_dir)
        num_batches = len(batches)

        if self.num_max_minibatches < num_batches:
            selected_batches = np.random.choice(range(num_batches),
                                                self.num_max_minibatches,
                                                replace=False)
        else:
            print("Number of batches not sufficient," +
                  " iterating over all the batches")
            selected_batches = np.random.permutation(range(num_batches))
        for idx in selected_batches:
            pairs = read_pairs(batches[idx])
            batch_els = self.load_frames_from_pairs(pairs)
            batch_els = map(torch.from_numpy, batch_els)
            X_batch1, X_batch2, y_batch = map(Variable, batch_els)
            yield X_batch1, X_batch2, y_batch


class FramesDataLoader(DataLoaderFromBatches):
    """
    This data loader constructs batches with frames, and not words (tokens).
    It can shuffle the tokens accross the whole dataset

    """

    def __init__(self, pairs_path, features_path, batch_size=100, randomize_dataset=True):
        """
        :parameter int batch_size: number of frames in a batch
        :param bool randomize_dataset: wether to shuffle all the frames between each epoch
        """
        super().__init__(pairs_path, features_path)
        self.randomize_dataset = randomize_dataset
        self.batch_size = batch_size

    def batch_iterator(self, train_mode=True):
        """
        This function is an iterator that will create batches for the whole dataset
        that was sampled by the sampler.
        Use it only if the sampler didn't create batches.

        It will randomize all your dataset before creating batches.
        Returns batches of the form (X1, X2, y)


        """
        if train_mode:
            batch_dir = os.path.join(self.pairs_path,
                                     'train_pairs')
        else:
            batch_dir = os.path.join(self.pairs_path,
                                     'dev_pairs')
        # read dataset
        dataset = os.path.join(batch_dir, 'dataset')
        pairs = read_pairs(dataset)

        # read all features
        self.load_features()

        X1, X2, y = self.load_frames_from_pairs(pairs)

        num_pair_tokens = len(X1)
        num_batches = num_pair_tokens // self.batch_size

        if num_batches == 0: num_batches = 1

        # randomized the dataset
        if self.randomize_dataset:
            perm = np.random.permutation(range(len(X1)))
        else:
            perm = np.arange(num_pair_tokens)  # identity
        X1 = X1[perm, :]
        X2 = X2[perm, :]
        y = y[perm]

        # make all batches
        x1_batches = np.array_split(X1, num_batches, axis=0)
        x2_batches = np.array_split(X2, num_batches, axis=0)
        y_batches = np.array_split(y, num_batches, axis=0)
        assert len(x1_batches) == len(x2_batches) == len(y_batches), "Number of batches does not correspond"

        # iterate
        for i in range(len(x1_batches)):
            X1_torch = Variable(torch.from_numpy(x1_batches[i]))
            X2_torch = Variable(torch.from_numpy(x2_batches[i]))
            y_torch = Variable(torch.from_numpy(y_batches[i]))
            yield X1_torch, X2_torch, y_torch


class MultiTaskDataLoader(FramesDataLoader):
    """
    This dataloader is optimized for the multitask siamese network
    """

    def __init__(self, pairs_path, features_path, fid2spk_file=None):

        super().__init__(pairs_path, features_path)
        self.fid2spk_file = fid2spk_file

    def batch_iterator(self, train_mode=True):
        """Build iteratior next batch from folder for a specific epoch
        Returns batches of the form (X1, X2, y_spk, y_phn)

        """

        if train_mode:
            batch_dir = os.path.join(self.pairs_path,
                                     'train_pairs')
        else:
            batch_dir = os.path.join(self.pairs_path,
                                     'dev_pairs')
        batches = Parse_Dataset(batch_dir)
        num_batches = len(batches)

        # read all features
        self.load_features()
        fid2spk = read_spkid_file(self.fid2spk_file)

        if self.num_max_minibatches < num_batches:
            selected_batches = np.random.choice(range(num_batches),
                                                self.num_max_minibatches,
                                                replace=False)
        else:
            print("Number of batches not sufficient," +
                  " iterating over all the batches")
            selected_batches = np.random.permutation(range(num_batches))
        for idx in selected_batches:
            pairs = read_pairs(batches[idx])
            batch_els = self.load_frames_from_pairs(
                pairs,
                fid2spk=fid2spk)
            batch_els = map(torch.from_numpy, batch_els)
            X_batch1, X_batch2, y_spk_batch, y_phn_batch = map(Variable,
                                                               batch_els)
            yield X_batch1, X_batch2, y_spk_batch, y_phn_batch
