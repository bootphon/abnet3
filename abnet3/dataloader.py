import numpy as np
import torch
from torch.autograd import Variable
import os
from collections import defaultdict
import random

from abnet3.utils import get_dtw_alignment, \
    Parse_Dataset, read_pairs, read_feats, \
    read_spkid_file, read_dataset, group_pairs, Features_Accessor

"""
This file contains several dataloaders.

DataLoaderFromBatches: Original method to load data.
It loads the batches of tokens created by the sampler, create the frame pairs,
and shuffles inside the batches.

FramesDataLoader: Reads a dataset composed of one file, loads all the frames in
memory, and shuffles across the
whole epoch. It then creates batches.

MultitaskDataLoader : default loader for multitask network

"""


class DataLoader:

    def batch_iterator(self, train_mode=True):
        """
        This function returns an iterator over all the batches of data
        :returns iterator
        """
        raise NotImplemented("You must implement batch iterator" +
                             " in DataLoader class.")

    def whoami(self):
        raise NotImplemented("You must implement whoami in DataLoader class")


class OriginalDataLoader(DataLoader):
    """
    Original method to load data.
    It loads the pairs file, created by the sampler
    create the frame pairs, and shuffles inside the batches.

    """

    TCL_DISTANCE_SAME = [1]  # see Synnaeve paper about Temporal Coherence Loss
    TCL_DISTANCES_DIFF = [15, 20, 25, 30]

    def __init__(self, pairs_path, features_path, num_max_minibatches=1000,
                 seed=None, batch_size=8, shuffle_between_epochs=False,
                 align_different_words=False,
                 tcl=0.0):
        """

        :param string pairs_path: path to dataset where the dev_pairs and
                                  train_pairs folders are
        :param features_path: path to feature file
        :param int num_max_minibatches: number of batches in each epoch
        :param int seed: for randomness
        :param bool align_different_words:
            If true, different words will be aligned along the diagonal.
            If false, the longest word will be truncated to match the length
            of the smallest word.
        :param tcl: temporal coherence loss percentage (0 <= tcl < 1)
        :param tcl_train_files: path to file listing the training wav items.
            This is used for temporal coherence loss. If None, all files will
            be considered.
        """
        assert 0 <= tcl < 1

        self.pairs_path = pairs_path
        self.features_path = features_path
        self.statistics_training = defaultdict(int)
        self.seed = seed
        self.num_max_minibatches = num_max_minibatches
        self.batch_size = batch_size
        self.features = None  # type: Features_Accessor
        self.shuffle_between_epochs = shuffle_between_epochs
        self.align_different_words = align_different_words
        self.tcl = tcl  # temporal coherence loss
        self.train_files = None # type: set
        self.pairs = {'train': None, 'dev': None}  # type: dict[str, list]

    def __getstate__(self):
        """used for pickle
        This function is used to remove the features in the state.
        They are very heavy (several GB)
        so we must remove them from the state before
        pickling and saving the network.
        """

        return (self.pairs_path,
                self.features_path,
                self.statistics_training,
                self.seed,
                self.num_max_minibatches,
                self.batch_size)

    def __setstate__(self, state):
        """
        As for __getstate__, this function is used to reconstruct the object
        from a pickled object.
        We have to reload the data since we didn't save the features
        and train / dev pairs.
        """
        (
            self.pairs_path,
            self.features_path,
            self.statistics_training,
            self.seed,
            self.num_max_minibatches,
            self.batch_size
        ) = state

        self.load_data()

    def whoami(self):
        return {
            'params': self.__getstate__(),
            'class_name': self.__class__.__name__
        }

    def load_data(self):
        """
        Load only once the features, and the pairs
        """
        if self.features is None:
            print("Loading features")
            features, align_features, feat_dim = read_feats(self.features_path)
            self.features = features

        if self.pairs['train'] is None:
            print("Loading word pairs")
            train_dir = os.path.join(self.pairs_path, 'train_pairs/dataset')
            self.pairs['train'] = read_dataset(train_dir)

        if self.pairs['dev'] is None:
            dev_dir = os.path.join(self.pairs_path, 'dev_pairs/dataset')
            self.pairs['dev'] = read_dataset(dev_dir)

        # analyse which are the train files
        self.train_files = list({pair[0] for pair in self.pairs['train']} |
                                {pair[3] for pair in self.pairs['train']})

    def get_token_feats(self, pairs):
        token_feats = {}
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            if (f1, s1, e1) not in token_feats:
                token_feats[f1, s1, e1] = self.features.get(f1, s1, e1)
            if (f2, s2, e2) not in token_feats:
                token_feats[f2, s2, e2] = self.features.get(f2, s2, e2)
        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            if (f1, s1, e1) not in token_feats:
                token_feats[f1, s1, e1] = self.features.get(f1, s1, e1)
            if (f2, s2, e2) not in token_feats:
                token_feats[f2, s2, e2] = self.features.get(f2, s2, e2)
        return token_feats

    def load_frames_from_pairs(self, pairs, seed=0, fid2spk=None):
        """Prepare a batch in Pytorch format based on a batch file
        :param pairs: list of pairs under the form
                      {'same': [pairs], 'diff': [pairs] }
        :param seed: randomness
        :param fid2spk:
            if None, will return X1, X2, y_phones
            If it is the spkid mapping, will return X1, X2, y_phones, y_speaker
        """
        # f are filenames, s are start times, e are end times
        token_feats = self.get_token_feats(pairs)

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
            except Exception:
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

            if self.align_different_words:
                # here we align the different words according to diagonal
                min_word = min((feat1, feat2), key=len)
                max_word = max((feat1, feat2), key=len)
                mapping = np.linspace(0, len(min_word) - 1,
                                      num=len(max_word))
                mapping = np.rint(mapping).astype(int)  # round to nearest integer
                min_word_mapped = min_word[mapping, :]
                word1 = max_word
                word2 = min_word_mapped
            else:
                word1 = feat1[:min(n1, n2), :]
                word2 = feat2[:min(n1, n2), :]
            X1.append(word1)
            X2.append(word2)
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
        X1, X2, y_phn = np.vstack(X1), np.vstack(X2), np.concatenate(y_phn)
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
        """Build iterator next batch from folder for a specific epoch
        This function can be used when the batches were already created
        by the sampler.

        If you use the sampler that didn't create batches, use the
        new_get_batches function
        Returns batches of the form (X1, X2, y)

        """
        # load features
        self.load_data()

        if train_mode:
            mode = 'train'
        else:
            mode = 'dev'
        pairs = self.pairs[mode]
        num_pairs = len(pairs)

        if self.shuffle_between_epochs:
            random.shuffle(pairs)

        # make batches
        sliced_indexes = range(0, num_pairs, self.batch_size)
        batches = [pairs[idx:idx + self.batch_size] for idx in sliced_indexes]
        num_batches = len(batches)

        if self.num_max_minibatches < num_batches:
            selected_batches = np.random.choice(range(num_batches),
                                                self.num_max_minibatches,
                                                replace=False)
        else:
            print("Number of batches not sufficient," +
                  " iterating over all the batches")
            selected_batches = np.random.permutation(range(num_batches))
        for batch_id in selected_batches:
            grouped_pairs = group_pairs(batches[batch_id])
            batch = self.load_frames_from_pairs(grouped_pairs)

            # add Temporal coherence loss
            if self.tcl > 0:
                batch = self.add_tcl_to_batch(batch)

            X1, X2, Y = batch
            X1, X2, Y = map(torch.from_numpy, [X1, X2, Y])
            X_batch1 = Variable(X1, volatile=not train_mode)
            X_batch2 = Variable(X2, volatile=not train_mode)
            y_batch = Variable(Y, volatile=not train_mode)
            yield X_batch1, X_batch2, y_batch

    def add_tcl_to_batch(self, batch):
        X1, X2, Y = batch
        num_pairs = len(Y)
        num_pairs_to_add = int((self.tcl * num_pairs) / (1 - self.tcl))
        X1_tcl, X2_tcl, Y_tcl = self.temporal_coherence_loss(num_pairs_to_add)
        X1 = np.vstack((X1, X1_tcl))
        X2 = np.vstack((X2, X2_tcl))
        Y = np.concatenate((Y, Y_tcl))
        return X1, X2, Y

    def temporal_coherence_loss(self, num_pairs):
        """
        As described in
            Dupoux, E., & Synnaeve, G. (2016).
            A Temporal Coherence Loss Function for
            Learning Unsupervised Acoustic Embeddings. SLTU.
        """
        X1, X2, Y = [], [], []
        pairs_per_iteration = len(self.TCL_DISTANCES_DIFF) + len(self.TCL_DISTANCE_SAME)
        for i in range(round(num_pairs /pairs_per_iteration)):
            files = list(self.features.features.keys())
            if self.train_files is not None:
                files = self.train_files
            f = random.choice(files)
            # pick random time in file
            file_features = self.features.features[f]
            t = random.choice(range(len(file_features) - max(self.TCL_DISTANCES_DIFF)))
            # "same" pairs
            for delta in self.TCL_DISTANCE_SAME:
                X1.append(file_features[t])
                X2.append(file_features[t + delta])
                Y.append(1)
            # "diff" pairs
            for delta in self.TCL_DISTANCES_DIFF:
                X1.append(file_features[t])
                X2.append(file_features[t + delta])
                Y.append(-1)

        return np.vstack(X1), np.vstack(X2), np.array(Y)


class TemporalCoherenceDataLoader(OriginalDataLoader):
    """
    This dataloader will load only temporal coherence pairs
    Which means for positive pairs : pairs that are close
    to one another, and for negative pairs, that are far.
    It won't use the sampled dataset for training, but
    it will use it for evaluation and early stopping.
    """

    def __init__(self, pairs_path, features_path, batch_size=500,
                 test_words_batch_size=8,
                 num_max_minibatches=1000):
        super().__init__(pairs_path, features_path,
                         num_max_minibatches=num_max_minibatches,
                         batch_size=test_words_batch_size)
        self.batch_size = batch_size

    def batch_iterator(self, train_mode=True):
        self.load_data()
        if train_mode:
            for _ in range(self.num_max_minibatches):
                batch = self.temporal_coherence_loss(num_pairs=self.batch_size)
                X1, X2, Y = map(torch.from_numpy, batch)
                X1 = Variable(X1, volatile=not train_mode)
                X2 = Variable(X2, volatile=not train_mode)
                Y = Variable(Y, volatile=not train_mode)
                yield X1, X2, Y
        else:
            yield from super(TemporalCoherenceDataLoader, self).batch_iterator(train_mode)


class FramesDataLoader(OriginalDataLoader):
    """
    This data loader constructs batches with frames, and not words (tokens).
    It can shuffle the tokens accross the whole dataset

    """

    def __init__(self, pairs_path, features_path,
                 batch_size=100, randomize_dataset=True, max_batches_per_epoch=None):
        """
        :parameter int batch_size: number of frames in a batch
        :param bool randomize_dataset: wether to shuffle all the frames
                                       between each epoch
        """
        super().__init__(pairs_path, features_path)
        self.randomize_dataset = randomize_dataset
        self.batch_size = batch_size
        self.token_features = {'train': None, 'dev': None}
        self.frame_pairs = {'train': None, 'dev': None}
        self.max_batches_per_epoch = max_batches_per_epoch

        if self.max_batches_per_epoch is not None:
            self.batch_position = 0  # batch position for train set

    def load_data(self):
        super(FramesDataLoader, self).load_data()

        if self.token_features['train'] is None:
            print("Loading all frames..", end='', flush=True)
            self.token_features['train'], self.frame_pairs['train'] = \
                self.load_all_frames(self.pairs['train'])
            print("Done. %s frame pairs in total." % len(self.frame_pairs['train']))

        if self.token_features['dev'] is None:
            self.token_features['dev'], self.frame_pairs['dev'] = \
                self.load_all_frames(self.pairs['dev'])

    def load_all_frames(self, pairs):
        """
        Loads all frames that appear in pair of tokens.
        It will return
            - a dictionnary token_feats that contains the list of frames for
            a given token (f, s, e)
            - a `frame` list, which contains the frame dataset :
            It is a list of (f1, s1, e1, index1, f2, s2, e2, index2, same)
            where
                -f1, f2 are the files
                - s1, s2, e1, e2 are the beginning and end of token
                - i1, i2 is the position in the token_feats dictionnary
                - same : value +1 or -1 depending if the two frames
                are the same or not.

        :param pairs:
            list of pairs under the form (f1, s1, e1, f2, s2, e2, same)
        """

        frames = []

        pairs = group_pairs(pairs)
        token_feats = self.get_token_feats(pairs)

        # get features for each same pair based on DTW alignment paths
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            if (s1 > e1) or (s2 > e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            try:
                path1, path2 = get_dtw_alignment(feat1, feat2)
            except Exception:
                continue

            for i1, i2 in zip(path1, path2):
                frames.append((f1, s1, e1, i1, f2, s2, e2, i2, 1))
            self.statistics_training['SameType'] += 1

        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            if (s1 > e1) or (s2 > e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]

            n1 = feat1.shape[0]
            n2 = feat2.shape[0]

            for i in range(min(n1, n2)):
                frames.append((f1, s1, e1, i, f2, s2, e2, i, -1))

            self.statistics_training['DiffType'] += 1

        np.random.shuffle(frames)
        return token_feats, frames

    def load_batch(self, frames, token_feats):

        X1, X2, Y = [], [], []

        for (f1, s1, e1, i1, f2, s2, e2, i2, y) in frames:
            feat1 = token_feats[f1, s1, e1][i1]
            feat2 = token_feats[f2, s2, e2][i2]
            X1.append(feat1)
            X2.append(feat2)
            Y.append(y)

        return np.vstack(X1), np.vstack(X2), np.array(Y)

    def batch_iterator(self, train_mode=True):
        """
        This function is an iterator that will create batches
        for the whole dataset
        that was sampled by the sampler.
        Use it only if the sampler didn't create batches.

        It will randomize all your dataset before creating batches.
        Returns batches of the form (X1, X2, y)

        """

        # read all features
        self.load_data()

        if train_mode:
            mode = 'train'
        else:
            mode = 'dev'

        frame_pairs = self.frame_pairs[mode]
        num_pairs = len(frame_pairs)
        num_batches = num_pairs // self.batch_size

        if num_batches == 0:
            num_batches = 1

        # choose which batches to run in this epoch
        if mode == 'dev' or self.max_batches_per_epoch is None:  # normal behaviour
            batch_ids = range(num_batches)
            if self.randomize_dataset:
                np.random.shuffle(frame_pairs)
        else:
            # we want to read only a subset of the dataset
            if self.batch_position >= num_batches:  # reset the count
                print("Arrived at the end of the dataset. Starting over.")
                if self.randomize_dataset:
                    np.random.shuffle(frame_pairs)
                self.batch_position = 0
            batch_ids = range(
                self.batch_position,
                min(self.batch_position + self.max_batches_per_epoch,
                    num_batches)
            )
            self.batch_position += self.max_batches_per_epoch

        for i in batch_ids:
            pairs_batch = frame_pairs[i*self.batch_size:
                                      i*self.batch_size + self.batch_size]
            X1, X2, y = self.load_batch(pairs_batch, self.token_features[mode])
            X1_torch = Variable(torch.from_numpy(X1), volatile=not train_mode)
            X2_torch = Variable(torch.from_numpy(X2), volatile=not train_mode)
            y_torch = Variable(torch.from_numpy(y), volatile=not train_mode)
            yield X1_torch, X2_torch, y_torch


class MultiTaskDataLoader(OriginalDataLoader):
    """
    This dataloader is optimized for the multitask siamese network
    """

    def __init__(self, pairs_path, features_path, fid2spk_file=None,
                 **kwargs):

        super().__init__(pairs_path, features_path, **kwargs)
        self.fid2spk_file = fid2spk_file

    def batch_iterator(self, train_mode=True):
        """Build iteratior next batch from folder for a specific epoch
        Returns batches of the form (X1, X2, y_spk, y_phn)

        """
        # load features
        self.load_data()

        if train_mode:
            mode = 'train'
        else:
            mode = 'dev'
        pairs = self.pairs[mode]
        num_pairs = len(pairs)

        # TODO : shuffle the pairs before creating batches
        # make batches
        sliced_indexes = range(0, num_pairs, self.batch_size)
        batches = [pairs[idx:idx + self.batch_size] for idx in sliced_indexes]
        num_batches = len(batches)

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
            pairs = group_pairs(batches[idx])
            batch_els = self.load_frames_from_pairs(
                pairs,
                fid2spk=fid2spk)
            batch_els = map(torch.from_numpy, batch_els)
            X_batch1, X_batch2, y_spk_batch, y_phn_batch = map(Variable,
                                                               batch_els)
            yield X_batch1, X_batch2, y_spk_batch, y_phn_batch
