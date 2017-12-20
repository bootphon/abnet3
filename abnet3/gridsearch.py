#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is composed of the different modules to experiment grid search
for ABnet3.

It will run a search optimization for the different parameters of the model

"""


class GridSearch(object):
    """Class Model for Grid search

        Parameters
        ----------
        input_file : String
            Path to clusters of words
        grid_sampler : dict
            dict for grid search parameters for sampler
        grid_trainer : dict
            dict for grid search parameters for trainer
        grid_model : dict
            dict for grid search parameters for model
        grid_loss : dict
            dict for grid search parameters for loss
        directory_output : String
            Path folder where train/dev pairs folder will be
        seed : int
            Seed
        num_jobs: int
            Number of jobs to use
        gpu_ids: List
            List of Gpu ids available for computation

    """
    def __init__(self, grid_sampler=None, grid_model=None, grid_loss=None,
                 grid_trainer=None, directory_output=None, seed=0,
                 num_jobs=1, gpu_ids=None):
        super(GridSearch, self).__init__()

    def whoami(self):
        raise NotImplementedError('Unimplemented whoami for class:',
                                  self.__class__.__name__)
