#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is composed of the different modules to experiment grid search
for ABnet3.

It will run a search optimization for the different parameters of the model

"""

import yaml
import faulthandler
import itertools
import re
import os
from path import Path
import time
import copy

from abnet3.sampler import *
from abnet3.loss import *
from abnet3.trainer import *
from abnet3.model import *
from abnet3.embedder import *
from abnet3.dataloader import *

faulthandler.enable()


class GridSearch(object):
    """Class Model for Grid search

        Parameters
        ----------
        input_file : String
            Path to yaml file for grid search
        num_jobs: int
            Number of jobs to use
        gpu_ids: List
            List of Gpu ids available for computation
        output_dir: String
            Path to directory for the outputs of the grid search

    """
    def __init__(self, input_file=None,
                 num_jobs=1, gpu_ids=None, output_dir=None):
        super(GridSearch, self).__init__()
        self.input_file = input_file
        self.num_jobs = num_jobs
        self.gpu_ids = gpu_ids
        self.output_dir = output_dir

    def whoami(self):
        raise NotImplementedError('Unimplemented whoami for class:',
                                  self.__class__.__name__)

    def parse_yaml_input_file(self):
        """Parse yaml input file for grid search

        """
        with open(self.input_file, 'r') as stream:
            try:
                self.params = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def run_single_experiment(self, single_experiment=None, gpu_id=0):
        """Build a single experiment from a dictionnary of parameters

        """
        assert single_experiment['features'], 'features properties missing'
        assert single_experiment['sampler'], 'sampler properties missing'
        assert single_experiment['trainer'], 'trainer properties missing'
        assert single_experiment['embedder'], 'embedder properties missing'
        assert single_experiment['model'], 'model properties missing'
        assert single_experiment['loss'], 'loss properties missing'

        sampler_prop = single_experiment['sampler']
        sampler = getattr(abnet3.sampler, sampler_prop['class'])(
            std_file=sampler_prop['std_file'],
            batch_size=sampler_prop['batch_size'],
            seed=sampler_prop['seed'],
            type_sampling_mode=sampler_prop['type_sampling_mode'],
            spk_sampling_mode=sampler_prop['spk_sampling_mode'],
            directory_output=sampler_prop['directory_output'],
            spkid_file=sampler_prop['spkid_file'],
            ratio_same_diff_spk=sampler_prop['ratio_same_diff_spk'],
            ratio_train_dev=sampler_prop['ratio_train_dev'],
            ratio_same_diff_type=sampler_prop['ratio_same_diff_type'],
            max_size_cluster=sampler_prop['max_size_cluster'],
            num_total_sampled_pairs=sampler_prop['num_total_sampled_pairs'],
            sample_batches=sampler_prop['sample_batches'],
        )
        import pdb
        pdb.set_trace()

        model_prop = single_experiment['model']
        model = SiameseNetwork(
            input_dim=input_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            p_dropout=p_dropout,
            activation_layer=activation_layer,
            batch_norm=batch_norm,
            output_path=output_network,
        )
        dataloader = OriginalDataLoader(
            pairs_path=directory_output,
            features_path=features_file,
            num_max_minibatches=num_max_minibatches,
            batch_size=batch_size
        )
        trainer = TrainerSiamese(
            network=network,
            loss=loss,
            dataloader=dataloader,
            cuda=cuda,
            feature_path=features_file,
            num_epochs=num_epochs,
            lr=learning_rate,
            patience=patience,
            num_max_minibatches=num_max_minibatches,
            optimizer_type=optimizer_type,
        )
        em = EmbedderSiamese(
               network,
               cuda=cuda,
               output_path=output_features,
               feature_path=features_file,
               network_path='network.pth',
        )

    def build_grid_experiments(self):
        """Extract the list of experiments to build the

        """
        self.parse_yaml_input_file()
        assert self.params['default_params'], 'Yaml not well formatted'
        assert self.params['grid_params'], 'Yaml not well formatted'

        default_params = self.params['default_params']
        grid_params = self.params['grid_params']
        grid_experiments = []
        current_exp = copy.deepcopy(default_params)
        for submodule, submodule_params in grid_params.items():
            for param, values in submodule_params.items():
                for value in values:
                    try:
                        current_exp[submodule][param] = value
                    except Exception as e:
                        current_exp[submodule] = {}
                        current_exp[submodule][param] = value
                    grid_experiments.append(current_exp)
                    current_exp = copy.deepcopy(default_params)
        return grid_experiments

    def run(self):
        """Run command to launch the grid search

        """
        grid_experiments = self.build_grid_experiments()
        self.run_single_experiment(single_experiment=grid_experiments[0])

    def make_html(self):
        """Build HTML outputs

        """


if __name__ == '__main__':
    grid = GridSearch(input_file='test/data/buckeye.yaml')
    grid.run()
    # import pdb
    # pdb.set_trace()
