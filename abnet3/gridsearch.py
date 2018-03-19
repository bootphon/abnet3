#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is composed of the different modules to experiment grid search
for ABnet3.

It will run a search optimization for the different parameters of the model

"""

import yaml
import faulthandler
import os
import time
import copy
import datetime
import argparse

from abnet3.sampler import *
from abnet3.loss import *
from abnet3.trainer import *
from abnet3.model import *
from abnet3.embedder import *
from abnet3.dataloader import *
from abnet3.features import *
import torch

faulthandler.enable()


class GridSearch(object):
    """Class Model for Grid search

        Parameters
        ----------
        input_file : String
            Path to yaml file for grid search
        num_jobs: int
            Number of jobs to use
        gpu_id: int
            Gpu id available for computation

    """
    def __init__(self, input_file=None,
                 num_jobs=1, gpu_ids=None,
                 date=None,
                 embed_only=False):
        self.input_file = input_file
        self.num_jobs = num_jobs
        self.gpu_ids = gpu_ids
        self.sampler_run = False
        self.features_run = False
        self.date = date
        self.embed_only = embed_only

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
        assert self.params['default_params']['pathname_experience'], \
            msg_yaml_error + 'pathname_experience'
        default_params = self.params['default_params']

        if 'grid_params' not in self.params:
            return [default_params]

        grid_params = self.params['grid_params']
        grid_experiments = []
        current_exp = copy.deepcopy(default_params)
        now = datetime.datetime.now().isoformat()

        if self.date is not None:
            now = self.date

        for submodule, submodule_params in grid_params.items():
            for param, values in submodule_params['arguments'].items():
                for value in values:
                    try:
                        current_exp[submodule]['arguments'][param] = value
                    except Exception as e:
                        current_exp[submodule]['arguments'] = {}
                        current_exp[submodule]['arguments'][param] = value
                    current_exp['pathname_experience'] = os.path.join(
                        current_exp['pathname_experience'], now,
                        param,
                        str(value)
                        )
                    grid_experiments.append(current_exp)
                    current_exp = copy.deepcopy(default_params)
        return grid_experiments

    def run_single_experiment(self, single_experiment=None, gpu_id=0):
        """Build a single experiment from a dictionnary of parameters

        """
        assert single_experiment['features'], 'features properties missing'
        assert single_experiment['sampler'], 'sampler properties missing'
        assert single_experiment['trainer'], 'trainer properties missing'
        assert single_experiment['embedder'], 'embedder properties missing'
        assert single_experiment['model'], 'model properties missing'
        assert single_experiment['loss'], 'loss properties missing'

        os.makedirs(single_experiment['pathname_experience'], exist_ok=True)

        features_prop = single_experiment['features']
        features_class = getattr(abnet3.features, features_prop['class'])
        arguments = features_prop['arguments']
        if 'output_path' not in arguments:
            arguments['output_path'] = os.path.join(
                single_experiment['pathname_experience'], 'features')
        features = features_class(**arguments)

        sampler_prop = single_experiment['sampler']
        sampler_class = getattr(abnet3.sampler, sampler_prop['class'])
        arguments = sampler_prop['arguments']
        if 'directory_output' not in arguments:
            arguments['directory_output'] = os.path.join(
                 single_experiment['pathname_experience'], 'pairs')
        sampler = sampler_class(**arguments)

        model_prop = single_experiment['model']
        model_class = getattr(abnet3.model, model_prop['class'])
        arguments = model_prop['arguments']
        arguments['output_path'] = os.path.join(
             single_experiment['pathname_experience'], 'network')
        model = model_class(**arguments)

        loss_prop = single_experiment['loss']
        loss_class = getattr(abnet3.loss, loss_prop['class'])
        arguments = loss_prop['arguments']
        loss = loss_class(**arguments)

        dataloader_prop = single_experiment['dataloader']
        dataloader_class = getattr(abnet3.dataloader, dataloader_prop['class'])
        arguments = dataloader_prop['arguments']
        arguments['pairs_path'] = sampler.directory_output
        arguments['features_path'] = features.output_path
        dataloader = dataloader_class(**arguments)

        trainer_prop = single_experiment['trainer']
        trainer_class = getattr(abnet3.trainer, trainer_prop['class'])
        arguments = trainer_prop['arguments']
        arguments['network'] = model
        arguments['loss'] = loss
        arguments['dataloader'] = dataloader
        arguments['log_dir'] = os.path.join(
             single_experiment['pathname_experience'],
             'logs')
        trainer = trainer_class(**arguments)

        embedder_prop = single_experiment['embedder']
        embedder_class = getattr(abnet3.embedder, embedder_prop['class'])
        arguments = embedder_prop['arguments']
        arguments['network'] = model
        if 'output_path' not in arguments:
            arguments['output_path'] = os.path.join(
                 single_experiment['pathname_experience'],
                 'embeddings.h5f')
        arguments['feature_path'] = features.output_path
        arguments['network_path'] = model.output_path + '.pth'
        embedder = embedder_class(**arguments)

        if self.embed_only:
            embedder.embed()
            return

        if features.run == 'never':
            pass
        if features.run == 'once' and self.features_run is False:
            features.generate()
            self.features_run = True
        if features.run == 'always':
            features.generate()
        else:
            pass

        if sampler.run == 'never':
            pass
        if sampler.run == 'once' and self.sampler_run is False:
            sampler.sample()
            self.sampler_run = True
        if sampler.run == 'always':
            sampler.sample()
        else:
            pass

        trainer.train()
        embedder.embed()

    def run(self):
        """Run command to launch the grid search

        """
        grid_experiments = self.build_grid_experiments()
        print('Start the grid search ...')
        for index in range(len(grid_experiments)):
            pathname_exp = grid_experiments[index]['pathname_experience']
            print('Starting exp {} : {}'.format(index, pathname_exp))
            self.run_single_experiment(
                single_experiment=grid_experiments[index]
                )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("experiments_file", type=str,
                           help="yaml file for the experiments")
    argparser.add_argument("--gpu_id", type=int, default=0,
                           help="Gpu id")
    argparser.add_argument("--num_jobs", type=int, default=1,
                           help="Not implemented yet")
    argparser.add_argument("-d", "--date", type=str,
                           help="Date to use to save the experiment")
    argparser.add_argument("--embed-only", action='store_true',
                           help="Run only the embedding (if the network is"
                                "already trained")

    args = argparser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    t1 = time.time()
    print("Start experiment")
    grid = GridSearch(input_file=args.experiments_file,
                      gpu_ids=args.gpu_id,
                      num_jobs=args.num_jobs,
                      date_field=args.date,
                      embed_only=args.embed_only)

    grid.run()
    print("The experiment took {} s ".format(time.time() - t1))


if __name__ == '__main__':
    main()
