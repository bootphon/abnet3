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
        dic_yaml = {'test_experiments_yaml': {
            'dataset': {
                'train': {
                    'feature_path': None},
                'test': {
                    'feature_path': None}}}}
        assert grid.grid_params == dic_yaml
