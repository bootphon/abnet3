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
from abnet3.integration import *
from abnet3.dataloader import DataLoader, FramesDataLoader, MultiTaskDataLoader
import numpy as np
import torch
import torch.optim as optim
import time
import pickle
import os
import matplotlib
import warnings
import copy
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboardX import SummaryWriter
from pathlib import Path


class TrainerBuilder:
    """Generic Trainer class for ABnet3

    """
    def __init__(self, network=None, loss=None,
                 num_epochs=200, patience=20,
                 optimizer_type='sgd', lr=0.001, momentum=0.9, cuda=True,
                 seed=0, dataloader=None, log_dir=None,
                 feature_generator=None):
        self.network = network
        self.loss = loss
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.momentum = momentum
        self.best_epoch = 0
        self.seed = seed
        self.cuda = cuda
        self.statistics_training = {}
        self.dataloader = dataloader
        self.feature_generator = feature_generator

        if cuda:
            self.loss.cuda()
            self.network.cuda()

        if log_dir is None:
            self.log_dir = Path('./runs/%s' % time.strftime('%m-%d-%Hh%M-%S'))
        else:
            self.log_dir = Path(log_dir) / ('%s' % time.strftime('%m-%d-%Hh%M-%S'))

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

    def params(self):
        params = copy.copy(self.__dict__)
        del params['dataloader']
        del params['feature_generator']

    def whoami(self):
        whoami = {
            'params': self.params(),
            'network': self.network.whoami(),
            'loss': self.loss.whoami(),
            'class_name': self.__class__.__name__,
            'dataloader': self.dataloader.whoami()
        }
        if self.feature_generator is not None:
            whoami['feature_generator'] = self.feature_generator.whoami()
        return whoami

    def save_whoami(self):
        pickle.dump(self.whoami(),
                    open(self.network.output_path+'.params', "wb"))

    def optimize_model(self, do_training=True):
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

        self.network.eval()
        self.network.save_network()

        train_writer = SummaryWriter(log_dir=str(self.log_dir / 'train_loss'))
        dev_writer = SummaryWriter(log_dir=str(self.log_dir / 'dev_loss'))

        _ = self.optimize_model(do_training=False)
        train_writer.add_scalar('loss', self.train_losses[-1], 0)
        dev_writer.add_scalar('loss', self.dev_losses[-1], 0)

        for key in self.statistics_training.keys():
            self.statistics_training[key] = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()

            dev_loss = self.optimize_model(do_training=True)

            # tensorboard logging
            train_writer.add_scalar('loss', self.train_losses[-1], epoch + 1)
            dev_writer.add_scalar('loss', self.dev_losses[-1], epoch + 1)

            if self.best_dev is None or dev_loss < self.best_dev:
                self.best_dev = dev_loss
                self.patience_dev = 0
                print('Saving best model so far, ' +
                      'epoch {}... '.format(epoch+1), end='', flush=True)
                self.network.save_network()
                self.save_whoami()
                print("Done.")
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
        assert type(self.network) == abnet3.model.SiameseNetwork

    def optimize_model(self, do_training=True):
        """Optimization model step for the Siamese network.

        """
        train_loss = 0.0
        dev_loss = 0.0
        self.network.train()

        for minibatch in self.dataloader.batch_iterator(train_mode=True):
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
        for minibatch in self.dataloader.batch_iterator(train_mode=False):
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
    def __init__(self, *args, **kwargs):
        super(TrainerSiameseMultitask, self).__init__(*args, **kwargs)
        assert type(self.network) == abnet3.model.SiameseMultitaskNetwork

    def optimize_model(self, do_training=True):
        """Optimization model step for the Siamese network with multitask.

        """
        train_loss = 0.0
        dev_loss = 0.0
        self.network.train()
        for minibatch in self.dataloader.batch_iterator(train_mode=True):
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
        for minibatch in self.dataloader.batch_iterator(train_mode=False):
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

class MultimodalTrainer(TrainerBuilder):
    """Multimodal Trainer class for ABnet3

    """
    def __init__(self, headstart=None, headstart_weight=None, *args, **kwargs):

        super(MultimodalTrainer, self).__init__(*args, **kwargs)
        assert type(self.dataloader) == abnet3.dataloader.MultimodalDataLoader

        if headstart:
            assert isinstance(self.network.integration_unit, BiWeightedScalarLearnt)
            self.headstart = headstart
            self.network.integration_unit.set_headstart_weight(headstart_weight)

    def cuda_all_modes(self, batch_list):
        cuda_on = []
        for mode in batch_list:
            cuda_on.append(mode.cuda())
        return cuda_on


    def optimize_model(self, do_training=True):
        """Optimization model step for the Multimodal Siamese network.

        """
        train_loss = 0.0
        dev_loss = 0.0
        self.network.train()

        if self.headstart == 0:
            self.network.integration_unit.start_training()
            print("Headstart ended")

        for X_list1, X_list2, y_batch in self.dataloader.batch_iterator(train_mode=True):
            if self.cuda:
                X_list1 = self.cuda_all_modes(X_list1)
                X_list2 = self.cuda_all_modes(X_list2)
                y_batch = y_batch.cuda()



            self.optimizer.zero_grad()
            emb_batch1, emb_batch2 = self.network(X_list1, X_list2)
            train_loss_value = self.loss(emb_batch1, emb_batch2, y_batch)
            if do_training:
                train_loss_value.backward()
                self.optimizer.step()
            else:
                self.num_batches_train += 1
            train_loss += train_loss_value.data[0]

        self.network.eval()
        for X_list1, X_list2, y_batch in self.dataloader.batch_iterator(train_mode=False):
            if self.cuda:
                X_list1 = self.cuda_all_modes(X_list1)
                X_list2 = self.cuda_all_modes(X_list2)
                y_batch = y_batch.cuda()

            if do_training:
                pass
            else:
                self.num_batches_dev += 1

            emb_batch1, emb_batch2 = self.network(X_list1, X_list2)
            dev_loss_value = self.loss(emb_batch1, emb_batch2, y_batch)
            dev_loss += dev_loss_value.data[0]

        self.train_losses.append(train_loss/self.num_batches_train)
        self.dev_losses.append(dev_loss/self.num_batches_dev)
        normalized_train_loss = train_loss/self.num_batches_train
        normalized_dev_loss = dev_loss/self.num_batches_dev

        self.pretty_print_losses(normalized_train_loss, normalized_dev_loss)

        if self.headstart > -1:
            self.headstart -= 1

        return dev_loss



if __name__ == '__main__':

    dataloader = MultiTaskDataLoader(pairs_path=None, features_path=None)

    sia = SiameseMultitaskNetwork(input_dim=280, num_hidden_layers_shared=2,
                                  hidden_dim=500,
                                  output_dim=100, p_dropout=0.,
                                  num_hidden_layers_spk=1,
                                  num_hidden_layers_phn=1,
                                  activation_layer='sigmoid',
                                  type_init='xavier_uni',
                                  batch_norm=False,
                                  output_path='/Users/rachine/abnet3/exp')
    coscos2_multi = weighted_loss_multi(loss=coscos2, weight=0.5)
    # sia.save_network()
    tra = TrainerSiameseMultitask(network=sia,
                                  dataloader=dataloader,
                                  loss=coscos2_multi, optimizer_type='adam',
                                  cuda=False)
