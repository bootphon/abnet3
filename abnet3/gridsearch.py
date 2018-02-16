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
import datetime

from abnet3.sampler import *
from abnet3.loss import *
from abnet3.trainer import *
from abnet3.model import *
from abnet3.embedder import *
from abnet3.dataloader import *
from abnet3.features import *

faulthandler.enable()


class GridSearchBuilder(object):
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
        super(GridSearchBuilder, self).__init__()
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

    def build_grid_experiments(self):
        """Extract the list of experiments to build the

        """
        self.parse_yaml_input_file()
        msg_yaml_error = 'Yaml not well formatted : '
        assert self.params['default_params'], msg_yaml_error + 'default_params'
        assert self.params['grid_params'], msg_yaml_error + 'grid_params'
        assert self.params['default_params']['pathname_experience'], \
            msg_yaml_error + 'pathname_experience'
        default_params = self.params['default_params']
        grid_params = self.params['grid_params']
        grid_experiments = []
        current_exp = copy.deepcopy(default_params)
        now = datetime.datetime.now()
        for submodule, submodule_params in grid_params.items():
            for param, values in submodule_params.items():
                for value in values:
                    try:
                        current_exp[submodule][param] = value
                    except Exception as e:
                        current_exp[submodule] = {}
                        current_exp[submodule][param] = value
                    current_exp['pathname_experience'] = os.path.join(
                        current_exp['pathname_experience'] + now.isoformat(),
                        param,
                        value
                        )
                    grid_experiments.append(current_exp)
                    current_exp = copy.deepcopy(default_params)
        return grid_experiments

    def run_single_experiment(self, single_experiment=None, gpu_id=0):
        """Build a single experiment from a dictionnary of parameters

        """
        raise NotImplementedError('Unimplemented run_single_experiment' +
                                  ' for class:',
                                  self.__class__.__name__)

    def run(self):
        """Run command to launch the grid search

        """
        grid_experiments = self.build_grid_experiments()
        print('Start the grid search ...')
        for index in len(grid_experiments):
            pathname_exp = grid_experiments[index]['pathname_experience']
            exp_name = 'Starting exp {} : {}'.format(index, pathname_exp)
            self.run_single_experiment(
                single_experiment=grid_experiments[index]
                )


class GridSearchSiamese(TrainerBuilder):
    """Siamese Trainer class for ABnet3

    """
    def __init__(self, *args, **kwargs):
        super(GridSearchBuilder, self).__init__(*args, **kwargs)

    def run_single_experiment(self, single_experiment=None, gpu_id=0):
        """Build a single experiment from a dictionnary of parameters

        """
        assert single_experiment['features'], 'features properties missing'
        assert single_experiment['sampler'], 'sampler properties missing'
        assert single_experiment['trainer'], 'trainer properties missing'
        assert single_experiment['embedder'], 'embedder properties missing'
        assert single_experiment['model'], 'model properties missing'
        assert single_experiment['loss'], 'loss properties missing'

        features_prop = single_experiment['features']
        features = getattr(abnet3.features, features_prop['class'])(
             files=features_prop['files'],
             output_path=features_prop['output_path'],
             load_mean_variance_path=features_prop['load_mean_variance_path'],
             save_mean_variance_path=features_prop['save_mean_variance_path'],
             vad_folder=features_prop['vad_folder'],
             n_filters=features_prop['n_filters'],
             method=features_prop['method'],
             normalization=features_prop['normalization'],
             norm_per_file=features_prop['norm_per_file'],
             stack=features_prop['stack'],
             nframes=features_prop['nframes'],
             deltas=features_prop['deltas'],
             deltasdeltas=features_prop['deltasdeltas']
        )

        sampler_prop = single_experiment['sampler']
        sampler = getattr(abnet3.sampler, sampler_prop['class'])(
            std_file=sampler_prop['std_file'],
            spk_list_file=sampler_prop['spk_list_file'],
            spkid_file=sampler_prop['spkid_file'],
            batch_size=sampler_prop['batch_size'],
            seed=sampler_prop['seed'],
            type_sampling_mode=sampler_prop['type_sampling_mode'],
            spk_sampling_mode=sampler_prop['spk_sampling_mode'],
            ratio_same_diff_spk=sampler_prop['ratio_same_diff_spk'],
            ratio_train_dev=sampler_prop['ratio_train_dev'],
            ratio_same_diff_type=sampler_prop['ratio_same_diff_type'],
            max_size_cluster=sampler_prop['max_size_cluster'],
            num_total_sampled_pairs=sampler_prop['num_total_sampled_pairs'],
            sample_batches=sampler_prop['sample_batches'],
            directory_output=os.path.join(
                 single_experiment['pathname_experience'], 'pairs')
        )

        model_prop = single_experiment['model']
        model = getattr(abnet3.model, model_prop['class'])(
            input_dim=model_prop['input_dim'],
            num_hidden_layers=model_prop['num_hidden_layers'],
            hidden_dim=model_prop['hidden_dim'],
            output_dim=model_prop['output_dim'],
            p_dropout=model_prop['p_dropout'],
            activation_layer=model_prop['activation_layer'],
            batch_norm=model_prop['batch_norm'],
            output_path=os.path.join(
                 single_experiment['pathname_experience'], 'network.pth')
        )

        loss_prop = single_experiment['loss']
        loss = getattr(abnet3.loss, loss_prop['class'])(
            avg=loss_prop['avg']
        )

        dataloader_prop = single_experiment['dataloader']
        dataloader = getattr(abnet3.dataloader, dataloader_prop['class'])(
            pairs_path=sampler.directory_output,
            features_path=features.output_path,
            num_max_minibatches=dataloader_prop['num_max_minibatches'],
            batch_size=dataloader_prop['batch_size'],
            seed=dataloader_prop['seed']
        )

        trainer_prop = single_experiment['trainer']
        trainer = getattr(abnet3.trainer, trainer_prop['class'])(
            network=model,
            loss=loss,
            dataloader=dataloader,
            cuda=trainer_prop['cuda'],
            num_epochs=trainer_prop['num_epochs'],
            lr=trainer_prop['lr'],
            patience=trainer_prop['patience'],
            optimizer_type=trainer_prop['optimizer_type'],
            log_dir=os.path.join(
                 single_experiment['pathname_experience'],
                 'logs')
        )

        embedder_prop = single_experiment['embedder']
        embedder = EmbedderSiamese(
               network=model,
               cuda=embedder_prop['cuda'],
               output_path=os.path.join(
                    single_experiment['pathname_experience'],
                    'embeddings.h5f'),
               feature_path=features.output_path,
               network_path=model.output_path,
        )

        if features.already_done:
            pass
        else:
            features.generate()

        if sampler.already_done:
            pass
        else:
            sampler.sample()

        trainer.train()
        embedder.embed()


if __name__ == '__main__':
    grid = GridSearchSiamese(input_file='test/data/buckeye.yaml')
    grid.run()
    # import pdb
    # pdb.set_trace()
