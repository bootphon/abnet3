#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pytest
from abnet3.gridsearch import *


class TestGridsearch:

    def test_parse_yaml_input_file(self):
        base_path = os.path.dirname(__file__)
        file_name = os.path.join(base_path, "data/empty.yaml")
        grid = GridSearch(input_file=file_name)

        grid.parse_yaml_input_file()
        dic_yaml = {'default_params': {
            'pathname_experience': '/empty_path',
            'features': {
                'already_done': True,
                'class': 'FeaturesGenerator'
            }
            },
            'grid_params': {
                'sampler': {
                    'type_sampling_mode': ['1', 'log', 'f3', 'f', 'f2']
                }
            }
            }

        assert grid.params == dic_yaml

    def test_build_grid_experiments(self):
        base_path = os.path.dirname(__file__)
        file_name = os.path.join(base_path, "data/empty.yaml")
        grid = GridSearch(input_file=file_name)

        grid_experiments = grid.build_grid_experiments()
        assert len(grid_experiments) == 5
        assert grid_experiments[0]['sampler']['type_sampling_mode'] == '1'
        assert grid_experiments[1]['sampler']['type_sampling_mode'] == 'log'


if __name__ == '__main__':
    pytest.main([__file__])
