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

    def build_single_experiment(self):
        """Build a single experiment from a dictionnary

        """

    def build_grid_experiments(self):
        """Extract the list of experiments to build the

        """
        self.parse_yaml_input_file()
        assert self.params['default_params'], 'Yaml not well formatted'
        assert self.params['grid_params'], 'Yaml not well formatted'

        default_params = self.params['default_params']
        grid_params = self.params['grid_params']
        grid_experiments = []
        for key in grid_params.keys():
            current_exp = copy.deepcopy(default_params)
            current_exp[key]
            import pdb
            pdb.set_trace()

    def run(self):
        """Run command to launch the grid search

        """

    def make_html(self):
        """Build HTML outputs

        """


if __name__ == '__main__':
    grid = GridSearch(input_file='test/data/buckeye.yaml')
    grid.build_grid_experiments()
    # import pdb
    # pdb.set_trace()
