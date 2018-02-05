#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is composed of the different modules to experiment grid search
for ABnet3.

It will run a search optimization for the different parameters of the model

"""

import yaml
import faulthandler
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

    """
    def __init__(self, input_file=None,
                 num_jobs=1, gpu_ids=None):
        super(GridSearch, self).__init__()
        self.input_file = input_file
        self.num_jobs = num_jobs
        self.gpu_ids = gpu_ids

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
        """Extract the several list of experiments

        """
        self.parse_yaml_input_file()
        experience_name = self.params.keys()
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    grid = GridSearch(input_file='test/data/buckeye.yaml')
    grid.build_grid_experiments()
    # import pdb
    # pdb.set_trace()
