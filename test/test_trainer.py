#!/usr/bin/env python3
# -*- coding: utf-8 -*
import os

from abnet3.trainer import TrainerSiamese
from abnet3.sampler import SamplerClusterSiamese
from abnet3.utils import read_dataset
from abnet3.model import SiameseNetwork
from abnet3.loss import coscos2

class TestTrainer:

    batch_size=5
    base_path = os.path.dirname(__file__)
    directory_output = os.path.join(base_path, "data/sampler_output/")
    dataset = os.path.join(directory_output, 'train_pairs/dataset')

    def get_trainer(self):
        network = SiameseNetwork(activation_layer='relu', input_dim=1, hidden_dim=1,
                                 num_hidden_layers=1, output_dim=1, cuda=False, loss=coscos2())
        sampler = SamplerClusterSiamese(directory_output=self.directory_output)
        trainer = TrainerSiamese(batch_size=self.batch_size, network=network, cuda=False, sampler=sampler)

        return trainer

    def test_new_get_batches(self):

        trainer = self.get_trainer()
        batches = trainer.create_batches_from_dataset()

        # test that all batches are of the good length
        assert all([len(batch['same']) + len(batch['diff']) == self.batch_size for batch in batches])

        # test that we have the good number of total samples (so the good number of batches)
        lines = read_dataset(self.dataset)
        assert (len(lines) // self.batch_size) * self.batch_size == sum(len(b['same']) + len(b['diff']) for b in batches)

        # test that all elements of batches are in dataset, in the correct category
        for b in batches:
            for f1, s1, e1, f2, s2, e2 in b['same']:
                assert (f1, s1, e1, f2, s2, e2, 'same') in lines
            for f1, s1, e1, f2, s2, e2 in b['diff']:
                assert (f1, s1, e1, f2, s2, e2, 'diff') in lines


