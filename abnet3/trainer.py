#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script is composed of the different modules for training neural
networks based on pairs of words, specific features from speech (usually
stacked filterbanks), a loss function, and a model.

It will generate models saved as .pth files to keep the weights and the
architecture of the best performance on the dev set.

"""

import abnet3
from abnet3.model import *
from abnet3.loss import *
from abnet3.sampler import *
from abnet3.utils import *
import numpy as np
import torch
import torch.optim as optim
import time
import pickle
import os
import matplotlib
import warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class TrainerBuilder:
    """Generic Trainer class for ABnet3

    """
    def __init__(self, sampler=None, network=None, loss=None,
                 feature_path=None,
                 num_epochs=200, patience=20, num_max_minibatches=1000,
                 optimizer_type='sgd', lr=0.001, momentum=0.9, cuda=True,
                 seed=0, batch_size=8, create_batches=False, randomize_dataset=True):
        # super(TrainerBuilder, self).__init__()
        self.sampler = sampler
        self.network = network
        self.loss = loss
        self.feature_path = feature_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_max_minibatches = num_max_minibatches
        self.lr = lr
        self.momentum = momentum
        self.best_epoch = 0
        self.seed = seed
        self.cuda = cuda
        self.statistics_training = {}
        self.batch_size = batch_size
        self.create_batches = create_batches
        self.randomize_dataset = randomize_dataset

        assert optimizer_type in ('sgd', 'adadelta', 'adam', 'adagrad',
                                  'RMSprop', 'LBFGS')
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=self.lr, momentum=self.momentum)
        if optimizer_type == 'adadelta':
            self.optimizer = optim.Adadelta(self.network.parameters(),
                                            lr=self.lr)
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=self.lr)
        if optimizer_type == 'adagrad':
            self.optimizer = optim.Adagrad(self.network.parameters(),
                                           lr=self.lr)
        if optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.network.parameters(),
                                           lr=self.lr)
        if optimizer_type == 'LBFGS':
            self.optimizer = optim.LBFGS(self.network.parameters(),
                                         lr=self.lr)
        if cuda:
            self.loss.cuda()
            self.network.cuda()

    def whoami(self):
        return {'params': self.__dict__,
                'network': self.network.whoami(),
                'loss': self.loss.whoami(),
                'sampler': self.sampler.whoami(),
                'class_name': self.__class__.__name__}

    def save_whoami(self):
        pickle.dump(self.whoami(),
                    open(self.network.output_path+'.params', "wb"))

    def optimize_model(self):
        """Optimization model step

        """
        raise NotImplementedError('Unimplemented optimize_model for class:',
                                  self.__class__.__name__)

    def train(self):
        """Train method to train the model

        """
        self.patience_dev = 0
        self.best_dev = None

        self.train_losses = []
        self.dev_losses = []
        self.num_batches_train = 0
        self.num_batches_dev = 0

        features, align_features, feat_dim = read_feats(self.feature_path)
        self.network.eval()
        self.network.save_network()

        _ = self.optimize_model(features, do_training=False)

        for key in self.statistics_training.keys():
            self.statistics_training[key] = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()

            dev_loss = self.optimize_model(features, do_training=True)

            if self.best_dev is None or dev_loss < self.best_dev:
                self.best_dev = dev_loss
                self.patience_dev = 0
                print('Saving best model so far, epoch {}'.format(epoch+1))
                self.network.save_network()
                self.save_whoami()
                self.best_epoch = epoch
            else:
                self.patience_dev += 1
                if self.patience_dev > self.patience:
                    print("No improvements after {} iterations, "
                          "stopping now".format(self.patience))
                    print('Finished Training')
                    break

        print('Saving best checkpoint network')

    def plot_train_erros(self):
        """Plot method to vizualize the train and dev errors

        """
        fig = plt.figure()
        x = range(len(self.train_losses))
        plt.plot(x, self.train_losses, 'r-')
        plt.plot(x, self.dev_losses, 'b+')
        fig.savefig(self.network.output_path+"_plot.pdf",
                    bbox_inches='tight')

    def plot_summary_statistics(self):
        """Summary statistics of the training

        """
        print(" ***** Statistics for the training step ***** ")
        for key in self.statistics_training.keys():
            stats = self.statistics_training[key]
            print(" Number of {} pairs seen: {} \t\t".format(key, stats))

    def pretty_print_losses(self, train_loss, dev_loss):
        """Print train and dev loss during training

        """
        print("  training loss:\t\t{:.6f}".format(train_loss))
        print("  dev loss:\t\t\t{:.6f}".format(dev_loss))


class TrainerSiamese(TrainerBuilder):
    """Siamese Trainer class for ABnet3

    """
    def __init__(self, *args, **kwargs):
        super(TrainerSiamese, self).__init__(*args, **kwargs)
        assert type(self.sampler) == abnet3.sampler.SamplerClusterSiamese
        assert type(self.network) == abnet3.model.SiameseNetwork

    def prepare_batch_from_pair_words(self, features, pairs,
                                      train_mode=True, seed=0):
        """Prepare a batch in Pytorch format based on a batch file
        :param pairs
        of the form
            {
                'same': [pairs],
                'diff': [pairs]
            }

        """

        # TODO should not be here, should be somewhere in the dataloader
        # TODO : Encapsulate X preparation in another function
        # TODO : Replace Numpy operation by Pytorch operation
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
        # get features for each same pair based on DTW alignment paths
        for f1, s1, e1, f2, s2, e2 in pairs['same']:
            if (s1 > e1) or (s2 > e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            try:
                path1, path2 = get_dtw_alignment(feat1, feat2)
            except Exception as e:
                continue
            try:
                self.statistics_training['SameType'] += 1
            except Exception as e:
                self.statistics_training['SameType'] = 1

            X1.append(feat1[path1, :])
            X2.append(feat2[path2, :])
            y.append(np.ones(len(path1)))

        for f1, s1, e1, f2, s2, e2 in pairs['diff']:
            if (s1 > e1) or (s2 > e2):
                continue
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
            n1 = feat1.shape[0]
            n2 = feat2.shape[0]
            X1.append(feat1[:min(n1, n2), :])
            X2.append(feat2[:min(n1, n2), :])
            y.append(-1*np.ones(min(n1, n2)))
            try:
                self.statistics_training['DiffType'] += 1
            except Exception as e:
                self.statistics_training['DiffType'] = 1

        # concatenate all features
        X1, X2, y = np.vstack(X1), np.vstack(X2), np.concatenate(y)
        np.random.seed(seed)
        n_pairs = len(y)
        ind = np.random.permutation(n_pairs)
        y = torch.from_numpy(y[ind])
        X1 = torch.from_numpy(X1[ind, :])
        X2 = torch.from_numpy(X2[ind, :])
        return X1, X2, y



    def get_batches(self, features, train_mode=True):
        """Build iteratior next batch from folder for a specific epoch
        This function can be used when the batches were already created
        by the sampler.

        If you use the sampler that didn't create batches, use the
        new_get_batches function

        """

        if train_mode:
            batch_dir = os.path.join(self.sampler.directory_output,
                                     'train_pairs')
        else:
            batch_dir = os.path.join(self.sampler.directory_output,
                                     'dev_pairs')

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
            bacth_els = self.prepare_batch_from_pair_words(
                    features, pairs , train_mode=train_mode)
            X_batch1, X_batch2, y_batch = map(Variable, bacth_els)
            yield X_batch1, X_batch2, y_batch

    def create_batches_from_dataset(self, train_mode=True):
        """
        This function will create batches for the whole dataset
        that was sampled by the sampler.
        Use it only if the sampler didn't create batches.

        It will randomize all your dataset before creating batches
        """
        if train_mode:
            batch_dir = os.path.join(self.sampler.directory_output,
                                     'train_pairs')
        else:
            batch_dir = os.path.join(self.sampler.directory_output,
                                     'dev_pairs')
        # read dataset
        dataset = os.path.join(batch_dir, 'dataset')
        pairs = read_dataset(dataset)
        num_pairs = len(pairs)
        print("We have in total %s pairs" % num_pairs)
        num_batches = num_pairs // self.batch_size

        # randomized the dataset
        if self.randomize_dataset:
            perm = np.random.permutation(range(num_pairs))
        else:
            perm = np.arange(num_pairs) # identity

        batches = []
        # group with batch
        for i in range(num_batches):
            indexes = perm[i*self.batch_size:(i+1)*self.batch_size]
            batch = [pairs[x] for x in indexes]
            grouped_batch = group_pairs(batch)
            batches.append(grouped_batch)
        return batches

    def new_get_batches(self, features, train_mode=True):
        batches = self.get_batches(features, train_mode=train_mode)
        for batch in batches:
            torch_batch = self.prepare_batch_from_pair_words(
                features, batch, train_mode=train_mode, seed=self.seed)
            X_batch1, X_batch2, y_batch = map(Variable, torch_batch)
            yield X_batch1, X_batch2, y_batch

    def optimize_model(self, features, do_training=True):
        """Optimization model step for the Siamese network.

        """
        train_loss = 0.0
        dev_loss = 0.0
        self.network.train()

        if self.create_batches:
            batch_iterator = self.new_get_batches
        else:
            batch_iterator = self.get_batches

        for minibatch in batch_iterator(features, train_mode=True):
            X_batch1, X_batch2, y_batch = minibatch
            if self.cuda:
                X_batch1 = X_batch1.cuda()
                X_batch2 = X_batch2.cuda()
                y_batch = y_batch.cuda()

            self.optimizer.zero_grad()
            emb_batch1, emb_batch2 = self.network(X_batch1, X_batch2)
            train_loss_value = self.loss(emb_batch1, emb_batch2, y_batch)
            if do_training:
                train_loss_value.backward()
                self.optimizer.step()
            else:
                self.num_batches_train += 1
            train_loss += train_loss_value.data[0]

        self.network.eval()
        for minibatch in batch_iterator(features, train_mode=False):
            X_batch1, X_batch2, y_batch = minibatch
            if self.cuda:
                X_batch1 = X_batch1.cuda()
                X_batch2 = X_batch2.cuda()
                y_batch = y_batch.cuda()

            if do_training:
                pass
            else:
                self.num_batches_dev += 1

            emb_batch1, emb_batch2 = self.network(X_batch1, X_batch2)
            dev_loss_value = self.loss(emb_batch1, emb_batch2, y_batch)
            dev_loss += dev_loss_value.data[0]

        self.train_losses.append(train_loss/self.num_batches_train)
        self.dev_losses.append(dev_loss/self.num_batches_dev)
        normalized_train_loss = train_loss/self.num_batches_train
        normalized_dev_loss = dev_loss/self.num_batches_dev

        self.pretty_print_losses(normalized_train_loss, normalized_dev_loss)
        return dev_loss


class TrainerSiameseMultitask(TrainerBuilder):
    """Siamese Trainer class for ABnet3 for multi task phn and spk

    """
    def __init__(self, fid2spk_file=None, *args, **kwargs):
        super(TrainerSiameseMultitask, self).__init__(*args, **kwargs)
        assert type(self.sampler) == abnet3.sampler.SamplerClusterSiamese
        assert type(self.network) == abnet3.model.SiameseMultitaskNetwork
        self.fid2spk_file = fid2spk_file

    def prepare_batch_from_pair_words(self, features, pairs_path,
                                      train_mode=True, seed=0, fid2spk=None):
        """Prepare a batch in Pytorch format based on a batch file

        """
        # f are filenames, s are start times, e are end times
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
            spk1, spk2 = fid2spk[f1], fid2spk[f2]
            if spk1 is spk2:
                y_spk.append(np.ones(len(path1)))
                try:
                    self.statistics_training['SameTypeSameSpk'] += 1
                except Exception as e:
                    self.statistics_training['SameTypeSameSpk'] = 1
            else:
                y_spk.append(-1*np.ones(len(path1)))
                try:
                    self.statistics_training['SameTypeDiffSpk'] += 1
                except Exception as e:
                    self.statistics_training['SameTypeDiffSpk'] = 1
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
            y_phn.append(-1*np.ones(min(n1, n2)))
            spk1, spk2 = fid2spk[f1], fid2spk[f2]
            if spk1 is spk2:
                y_spk.append(np.ones(min(n1, n2)))
                try:
                    self.statistics_training['DiffTypeSameSpk'] += 1
                except Exception as e:
                    self.statistics_training['DiffTypeSameSpk'] = 1
            else:
                y_spk.append(-1*np.ones(min(n1, n2)))
                try:
                    self.statistics_training['DiffTypeDiffSpk'] += 1
                except Exception as e:
                    self.statistics_training['DiffTypeDiffSpk'] = 1

        # concatenate all features
        X1, X2 = np.vstack(X1), np.vstack(X2)
        y_phn, y_spk = np.concatenate(y_phn), np.concatenate(y_spk)
        np.random.seed(seed)
        assert len(y_phn) == len(y_spk), 'not same number of labels...'
        n_pairs = len(y_phn)
        ind = np.random.permutation(n_pairs)
        y_phn = torch.from_numpy(y_phn[ind])
        y_spk = torch.from_numpy(y_spk[ind])
        X1 = torch.from_numpy(X1[ind, :])
        X2 = torch.from_numpy(X2[ind, :])
        return X1, X2, y_spk, y_phn

    def get_batches(self, features, train_mode=True):
        """Build iteratior next batch from folder for a specific epoch

        """

        if train_mode:
            batch_dir = os.path.join(self.sampler.directory_output,
                                     'train_pairs')
        else:
            batch_dir = os.path.join(self.sampler.directory_output,
                                     'dev_pairs')

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
            bacth_els = self.prepare_batch_from_pair_words(
                    features, batches[idx], train_mode=train_mode,
                    fid2spk=read_spkid_file(self.fid2spk_file))

            X_batch1, X_batch2, y_spk_batch, y_phn_batch = map(Variable,
                                                               bacth_els)
            yield X_batch1, X_batch2, y_spk_batch, y_phn_batch

    def optimize_model(self, features, do_training=True):
        """Optimization model step for the Siamese network with multitask.

        """
        train_loss = 0.0
        dev_loss = 0.0
        self.network.train()
        for minibatch in self.get_batches(features, train_mode=True):
            X_batch1, X_batch2, y_spk_batch, y_phn_batch = minibatch
            if self.cuda:
                X_batch1 = X_batch1.cuda()
                X_batch2 = X_batch2.cuda()
                y_spk_batch = y_spk_batch.cuda()
                y_phn_batch = y_phn_batch.cuda()

            self.optimizer.zero_grad()
            emb = self.network(X_batch1, X_batch2)
            emb_spk1, emb_phn1, emb_spk2, emb_phn2 = emb
            train_loss_value = self.loss(emb_spk1, emb_phn1,
                                         emb_spk2, emb_phn2,
                                         y_spk_batch, y_phn_batch)
            if do_training:
                train_loss_value.backward()
                self.optimizer.step()
            else:
                self.num_batches_train += 1
            train_loss += train_loss_value.data[0]

        self.network.eval()
        for minibatch in self.get_batches(features, train_mode=False):
            X_batch1, X_batch2, y_spk_batch, y_phn_batch = minibatch
            if self.cuda:
                X_batch1 = X_batch1.cuda()
                X_batch2 = X_batch2.cuda()
                y_spk_batch = y_spk_batch.cuda()
                y_phn_batch = y_phn_batch.cuda()

            if do_training:
                pass
            else:
                self.num_batches_dev += 1

            emb = self.network(X_batch1, X_batch2)
            emb_spk1, emb_phn1, emb_spk2, emb_phn2 = emb
            dev_loss_value = self.loss(emb_spk1, emb_phn1,
                                       emb_spk2, emb_phn2,
                                       y_spk_batch, y_phn_batch)
            dev_loss += dev_loss_value.data[0]

        self.train_losses.append(train_loss/self.num_batches_train)
        self.dev_losses.append(dev_loss/self.num_batches_dev)
        normalized_train_loss = train_loss/self.num_batches_train
        normalized_dev_loss = dev_loss/self.num_batches_dev

        self.pretty_print_losses(normalized_train_loss, normalized_dev_loss)
        return dev_loss


if __name__ == '__main__':

    sia = SiameseMultitaskNetwork(input_dim=280, num_hidden_layers_shared=2,
                                  hidden_dim=500,
                                  output_dim=100, p_dropout=0.,
                                  num_hidden_layers_spk=1,
                                  num_hidden_layers_phn=1,
                                  activation_layer='sigmoid',
                                  type_init='xavier_uni',
                                  batch_norm=False,
                                  output_path='/Users/rachine/abnet3/exp',
                                  cuda=False)
    sam = SamplerClusterSiamese(already_done=True, directory_output=None)
    coscos2_multi = weighted_loss_multi(loss=coscos2, weight=0.5)
    # sia.save_network()
    tra = TrainerSiameseMultitask(sampler=sam, network=sia,
                                  loss=coscos2_multi, optimizer_type='adam',
                                  cuda=False)
