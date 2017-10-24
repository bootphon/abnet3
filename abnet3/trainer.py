#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:24:59 2017

@author: Rachid Riad
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class TrainerBuilder:
    """Generic Trainer class for ABnet3
    
    """
    def __init__(self, sampler, network, loss, feature_path=None,
                 num_epochs=200, patience=20, num_max_minibatches=1000,
                 optimizer_type='SGD', lr=0.001, momentum=0.9, cuda=True,
                 seed=0):
#        super(TrainerBuilder, self).__init__()
        self.sampler = sampler
        self.network = network
        self.loss = loss
        self.feature_path = feature_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_max_minibatches = num_max_minibatches
        self.lr = lr
        self.momentum = momentum
        self.best_epoch = None
        self.seed = seed
        self.cuda = cuda
        assert optimizer_type in ('sgd', 'adadelta','adam','adagrad')
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        if optimizer_type == 'adadelta':
            self.optimizer = optim.Adadelta(self.network.parameters())
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.network.parameters())
        if optimizer_type == 'adagrad':
            self.optimizer = optim.Adagrad(self.network.parameters())
        if cuda:
            self.loss.cuda()
            self.network.cuda()
            
    def whoami(self):
        return {'params':self.__dict__,
                'network':self.network.whoami(),
                'loss':self.loss.whoami(),
                'sampler':self.sampler.whoami(),
                'class_name': self.__class__.__name__}
        
    def save_whoami(self):
        pickle.dump(self.whoami(),  
            open(self.network.output_path+'.params',"wb" ))
        
    def optimize_model(self):
        """Optimization model step
        """
        raise NotImplementedError('Unimplemented optimize_model for class:',
                  self.__class__.__name__)
        
    def train(self):
        """Train function 
    
        """
        raise NotImplementedError('Unimplemented train for class:',
                          self.__class__.__name__)

    def plot_train_erros(self):
        """Plot function for training losses 
    
        """
        raise NotImplementedError('Unimplemented plot_train_erros for class:',
                          self.__class__.__name__)

       
class TrainerSiamese(TrainerBuilder):
    """Siamese Trainer class for ABnet3
    
    """
    def __init__(self,*args, **kwargs):
        super(TrainerSiamese, self).__init__(*args, **kwargs) 
        assert type(self.sampler) == abnet3.sampler.SamplerClusterSiamese
        assert type(self.network) == abnet3.model.SiameseNetwork
    
    def prepare_batch_from_pair_words(self, features, pairs_path, train_mode=True,seed=0):
        """Prepare a batch in Pytorch format based on a batch file
        
        """
        

        #TODO should not be here, should be somewhere in the dataloader 
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
        y = torch.from_numpy(y[ind])
        X1 = torch.from_numpy(X1[ind,:])
        X2 = torch.from_numpy(X2[ind,:])
        return X1, X2, y
    
    def get_batches(self, features, train_mode=True):
        """Build iteratior next batch from folder for a specific epoch
        
        """
        
        if train_mode:
            batch_dir = os.path.join(self.sampler.directory_output,'train_pairs')
        else:
            batch_dir = os.path.join(self.sampler.directory_output,'dev_pairs')
            
        batches = Parse_Dataset(batch_dir)
        num_batches = len(batches)
        if self.num_max_minibatches<num_batches:
            selected_batches = np.random.choice(range(num_batches), self.num_max_minibatches, replace=False)
        else:
            print("Number of batches not sufficient, iterating over all the batches")
            selected_batches = np.random.permutation(range(num_batches))
        for idx in selected_batches:
            X_batch1, X_batch2, y_batch = self.prepare_batch_from_pair_words(features, batches[idx],  train_mode=train_mode)
            yield Variable(X_batch1, requires_grad=False), Variable(X_batch2, requires_grad=False), Variable(y_batch, requires_grad=False)
        
        
    def train(self):
        """Train method to train the model
        
        """
        patience_dev = 0
        best_dev = None
        
        self.train_losses = []
        self.dev_losses = []
        
        features, align_features, feat_dim = read_feats(self.feature_path)
        train_loss = 0.0
        dev_loss = 0.0
        self.network.eval()
        self.network.save_network()
        
        self.network.train()
        num_batches_train = 0
        for minibatch in self.get_batches(features, train_mode=True):
            #TODO refactor here for a step function based on specific loss
            # enable generic train
            
            X_batch1, X_batch2, y_batch = minibatch
            if self.cuda:
                X_batch1 = X_batch1.cuda()
                X_batch2 = X_batch2.cuda()
                y_batch  = y_batch.cuda()
            emb_batch1, emb_batch2 = self.network.forward(X_batch1,X_batch2)
            train_loss_value = self.loss.forward(emb_batch1, emb_batch2, y_batch)
            train_loss += train_loss_value.data[0]
            
            num_batches_train +=1
            
        self.train_losses.append(train_loss)
        self.network.eval()   
        num_batches_dev = 0
        for minibatch in self.get_batches(features, train_mode=False):
            X_batch1, X_batch2, y_batch = minibatch
            if self.cuda:
                X_batch1 = X_batch1.cuda()
                X_batch2 = X_batch2.cuda()
                y_batch  = y_batch.cuda()
            
            emb_batch1, emb_batch2 = self.network.forward(X_batch1,X_batch2)
            dev_loss_value = self.loss.forward(emb_batch1, emb_batch2, y_batch)
            dev_loss += dev_loss_value.data[0]
                
            num_batches_dev += 1
        
        self.dev_losses.append(dev_loss)
        
        print("  training loss:\t\t{:.6f}".format(train_loss/num_batches_train))
        print("  dev loss:\t\t\t{:.6f}".format(dev_loss/num_batches_dev))
        
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            dev_loss = 0.0
            start_time = time.time()
            
            self.network.train()
            for minibatch in self.get_batches(features, train_mode=True):
                #TODO refactor here for a step function based on specific loss
                # enable generic train
                X_batch1, X_batch2, y_batch = minibatch
                if self.cuda:
                    X_batch1 = X_batch1.cuda()
                    X_batch2 = X_batch2.cuda()
                    y_batch  = y_batch.cuda()
                
                self.optimizer.zero_grad()
                emb_batch1, emb_batch2 = self.network.forward(X_batch1,X_batch2)
                train_loss_value = self.loss.forward(emb_batch1, emb_batch2, y_batch)
                train_loss_value.backward()
                self.optimizer.step()
                train_loss += train_loss_value.data[0]
                
            self.train_losses.append(train_loss)
            
            self.network.eval()
            for minibatch in self.get_batches(features, train_mode=False):
                X_batch1, X_batch2, y_batch = minibatch
                if self.cuda:
                    X_batch1 = X_batch1.cuda()
                    X_batch2 = X_batch2.cuda()
                    y_batch  = y_batch.cuda()
                
                emb_batch1, emb_batch2 = self.network.forward(X_batch1,X_batch2)
                dev_loss_value = self.loss.forward(emb_batch1, emb_batch2, y_batch)
                dev_loss += dev_loss_value.data[0]
            
            self.dev_losses.append(dev_loss)
            
            print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, self.num_epochs, time.time() - start_time))
            
            print("  training loss:\t\t{:.6f}".format(train_loss/num_batches_train))
            print("  dev loss:\t\t\t{:.6f}".format(dev_loss/num_batches_dev))
            if best_dev == None or dev_loss < best_dev:
                best_dev = dev_loss
                patience_dev = 0
                print('Saving best model so far, epoch {}'.format(epoch+1))
                self.network.save_network()
                self.save_whoami()
                self.best_epoch  = epoch
            else:
                patience_dev += 1
                if patience_dev > self.patience:
                    print("No improvements after {} iterations, "
                      "stopping now".format(self.patience))
                    print('Finished Training')
                    break
        
        print('Saving best checkpoint network')
        
        self.network.load_network(network_path = self.network.output_path+'.pth')
        print('The best epoch is the {}-th'.format(self.best_epoch))
        print('The best train is the {}'.format(self.train_losses[self.best_epoch]))
        print('The best dev is the {}'.format(self.dev_losses[self.best_epoch]))
        print('Still Training but no more patience.')
        print('Finished Training')
    
    def plot_train_erros(self):
        """Plot method to vizualize the train and dev errors
        
        """
        fig = plt.figure()        
#        plt.gca().set_color_cycle(['red', 'blue'])
        x = range(len(self.train_losses))
        plt.plot(x,self.train_losses,'r-')
        plt.plot(x,self.dev_losses,'b+')
        fig.savefig(self.network.output_path+ "_plot.pdf",
                    bbox_inches='tight')
        
        
        
    
if __name__ == '__main__':
    
    sia = SiameseNetwork(input_dim=3,num_hidden_layers=2,hidden_dim=10,
                     output_dim=19,p_dropout=0.1,
                     activation_layer='sigmoid',
                     batch_norm=True, output_path='/home/rachine/abnet3/exp')
    sam = SamplerClusterSiamese(already_done=True, directory_output=None)
    loss = coscos2()
    sia.save_network()
    tra = TrainerSiamese(sam,sia,loss, optimizer_type='adam')

    

        
        
        