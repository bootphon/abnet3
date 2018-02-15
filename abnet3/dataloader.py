import numpy as np
import torch
from torch.autograd import Variable
import os
from collections import defaultdict

from abnet3.utils import get_dtw_alignment, \
    Parse_Dataset, read_pairs, read_feats, \
    read_spkid_file, read_dataset, group_pairs

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

    def __init__(self, pairs_path, features_path, num_max_minibatches=1000,
                 seed=None, batch_size=8):
        """

        :param string pairs_path: path to dataset where the dev_pairs and
                                  train_pairs folders are
        :param features_path: path to feature file
        :param int num_max_minibatches: number of batches in each epoch
        :param int seed: for randomness
        """
        self.pairs_path = pairs_path
        self.features_path = features_path
        self.statistics_training = defaultdict(int)
        self.seed = seed
        self.num_max_minibatches = num_max_minibatches
        self.batch_size = batch_size
        self.features = None
        self.train_pairs = None
        self.dev_pairs = None

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
            features, align_features, feat_dim = read_feats(self.features_path)
            self.features = features

        if self.train_pairs is None:
            train_dir = os.path.join(self.pairs_path, 'train_pairs/dataset')

            self.train_pairs = read_dataset(train_dir)

        if self.dev_pairs is None:
            dev_dir = os.path.join(self.pairs_path, 'dev_pairs/dataset')
            self.dev_pairs = read_dataset(dev_dir)

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
            pairs = self.train_pairs
        else:
            pairs = self.dev_pairs

        num_pairs = len(pairs)

        # TODO : shuffle the pairs before creating batches
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
            batch_els = self.load_frames_from_pairs(grouped_pairs)
            batch_els = map(torch.from_numpy, batch_els)
            X_batch1, X_batch2, y_batch = map(Variable, batch_els)
            yield X_batch1, X_batch2, y_batch


class FramesDataLoader(OriginalDataLoader):
    """
    This data loader constructs batches with frames, and not words (tokens).
    It can shuffle the tokens accross the whole dataset

    """

    def __init__(self, pairs_path, features_path,
                 batch_size=100, randomize_dataset=True):
        """
        :parameter int batch_size: number of frames in a batch
        :param bool randomize_dataset: wether to shuffle all the frames
                                       between each epoch
        """
        super().__init__(pairs_path, features_path)
        self.randomize_dataset = randomize_dataset
        self.batch_size = batch_size
        self.X1 = None

    def batch_iterator(self, train_mode=True):
        """
        This function is an iterator that will create batches
        for the whole dataset
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
        self.load_data()

        if self.X1 is None:
            self.X1, self.X2, self.y = self.load_frames_from_pairs(pairs)

        num_pair_frames = len(self.X1)
        num_batches = num_pair_frames // self.batch_size

        if num_batches == 0:
            num_batches = 1

        # randomized the dataset
        if self.randomize_dataset:
            perm = np.random.permutation(range(num_pair_frames))
        else:
            perm = np.arange(num_pair_frames)  # identity
        X1 = self.X1[perm, :]
        X2 = self.X2[perm, :]
        y = self.y[perm]

        # make all batches
        x1_batches = np.array_split(X1, num_batches, axis=0)
        x2_batches = np.array_split(X2, num_batches, axis=0)
        y_batches = np.array_split(y, num_batches, axis=0)
        msg_error = "Number of batches does not correspond"
        assert len(x1_batches) == len(x2_batches) == len(y_batches), msg_error

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
        self.load_data()
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

class MultimodalDataLoader(OriginalDataLoader):
    """
    Class to manage multiple inputs, extract features
    from features file and provide iterator for multimodal
    training of the siamese network
    """

    def __init__(self, pairs_path, features_path, num_max_minibatches=1000, seed=None, batch_size=8):
        """

        :param string pairs_path: path to dataset where the dev_pairs and train_pairs folders are
        :param features_paths: list of paths from multiple inputs, this turns the OriginalDataLoader
                               features_path parameter into a list

        """
        super().__init__(pairs_path, features_path)
        self.features_dict = None

        #TODO: label different modes for later analysis


    def __getstate__(self):
        """used for pickle"""

        #TODO: implement
        pass

    def __setstate__(self, state):
        """used for pickle"""

        #TODO: implement
        pass

    def whoami(self):

        #TODO: implement
        pass

    def check_consistency(self, features, deep=True):
        """
        This method checks that the pairs and features are
        consistent between each other, meaning they have the
        same items and can be used together

        :param features: list of features to be used
        :param deep: when True, a more extensive
        """

        #TODO: implement, for now consistent data is assumed.
        pass

    def load_data(self):
        """
        Load pairs and features
        """

        if self.features_dict is None:
            self.features_dict = {}
            for path in self.features_path:
                self.features_dict[path], _ , _ = read_feats(path)

        if self.train_pairs is None:
            train_dir = os.path.join(self.pairs_path, 'train_pairs/dataset')

            self.train_pairs = read_dataset(train_dir)

        if self.dev_pairs is None:
            dev_dir = os.path.join(self.pairs_path, 'dev_pairs/dataset')
            self.dev_pairs = read_dataset(dev_dir)


    def batch_iterator(self, train_mode=True):
        """
        Build iterator next batch from folder for a specific epoch
        This function can be used when the batches were already created
        by the sampler.

        :param train: boolean, indicates if the pairs should be extracted from
                      the train set (if True) or the dev set (if False)

        Returns batches of the form (X1array, X2array, y), where X1list and
        X2list are lists of torch variables, each one corresponding
        to the different modes representing the same phenomena,
        and y is a torch variable which contains same/different info, which is
        the same for every mode
        """

        #TODO: support unsampled batches

        #load pairs and features
        self.load_data()

        if train_mode:
            pairs = self.train_pairs
        else:
            pairs = self.dev_pairs

        num_pairs = len(pairs)

        # TODO : shuffle the pairs before creating batches
        # make batches
        batches = [pairs[i:i+self.batch_size] for i in range(0, num_pairs, self.batch_size)]
        num_batches = len(batches)

        if self.num_max_minibatches < num_batches:
            selected_batches = np.random.choice(range(num_batches),
                                                self.num_max_minibatches,
                                                replace=False)
        else:
            print("Number of batches not sufficient," +
                  " iterating over all the batches")
            selected_batches = np.random.permutation(range(num_batches))

        #Yield batches
        for batch_id in selected_batches:
            grouped_pairs = group_pairs(batches[batch_id])
            X1list, X2list = [], []
            i = 1
            for path in self.features_path:
                self.features = self.features_dict[path]
                batch_els = self.load_frames_from_pairs(grouped_pairs)
                for _ in batch_els:
                    print(np.shape(_))
                batch_els = map(torch.from_numpy, batch_els)
                X_batch1, X_batch2, y_batch = map(Variable, batch_els)
                X1list.append(X_batch1)
                X2list.append(X_batch2)

                print("Input {} X1 size {}".format(i, X_batch1.size()))
                print("Input {} X2 size {}".format(i, X_batch2.size()))
                i += 1
            yield X1list, X2list, y_batch
