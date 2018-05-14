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
import argparse

import abnet3.features
import abnet3.model
import abnet3.embedder

import torch

faulthandler.enable()


class EmbedCLI(object):
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
    def __init__(self, yaml_file=None,
                 weights=None, input_features=None):
        self.yaml_file = yaml_file
        self.sampler_run = False
        self.features_run = False
        self.weights = weights
        self.input_features = input_features

        self.test_files = []

    def parse_yaml_input_file(self):
        """Parse yaml input file for grid search

        """
        with open(self.yaml_file, 'r') as stream:
            self.params = yaml.load(stream)

    def run_embedding(self, single_experiment=None):

        if self.input_features is None:
            features_prop = single_experiment['features']
            features_class = getattr(abnet3.features, features_prop['class'])
            arguments = features_prop['arguments']
            if 'output_path' not in arguments:
                arguments['output_path'] = os.path.join(
                    single_experiment['pathname_experience'], 'features')
            features = features_class(**arguments)
            if not os.path.exists(arguments['output_path']):
                features.generate()
            self.input_features = arguments['output_path']
            print("Using default features : %s" % self.input_features)

        model_prop = single_experiment['model']
        model_class = getattr(abnet3.model, model_prop['class'])
        arguments = model_prop['arguments']
        arguments['output_path'] = os.path.join(
             single_experiment['pathname_experience'], 'network')
        model = model_class(**arguments)

        embedder_prop = single_experiment['embedder']
        embedder_class = getattr(abnet3.embedder, embedder_prop['class'])
        arguments = embedder_prop['arguments']
        arguments['network'] = model
        if 'output_path' not in arguments:
            arguments['output_path'] = os.path.join(
                 single_experiment['pathname_experience'],
                 'embeddings.h5f')

        arguments['feature_path'] = self.input_features
        if self.weights is not None:
            print("using weights in %s" % self.weights)
            arguments['network_path'] = self.weights
        else:
            arguments['network_path'] = model.output_path + '.pth'
        embedder = embedder_class(**arguments)

        embedder.embed()

        # # embed test features
        # if self.test_files:
        #     for file in self.test_files:
        #         test_wavs = file["files"]
        #         name = file["name"]
        #         if "features" in file:
        #             test_features = file["features"]
        #         else:
        #             test_features = os.path.join(
        #                     single_experiment['pathname_experience'],
        #                     'test-{name}'.format(name=name))
        #         vad_file = None
        #         if "vad_file" in file:
        #             vad_file = file["vad_file"]
        #
        #         if not os.path.exists(test_features):
        #             # create test features
        #             print("Creating test features for %s at path %s" %
        #                   (name, test_features))
        #             features_prop = single_experiment['features']
        #             features_class = getattr(abnet3.features,
        #                                      features_prop['class'])
        #             arguments = features_prop['arguments']
        #             arguments["files"] = test_wavs
        #             arguments["vad_file"] = vad_file
        #             arguments["output_path"] = test_features
        #             features = features_class(**arguments)
        #
        #             features.generate()
        #
        #         # run embedding
        #         embedder_prop = single_experiment['embedder']
        #         embedder_class = getattr(abnet3.embedder, embedder_prop['class'])
        #         arguments = embedder_prop['arguments']
        #         arguments['network'] = model
        #         output_path = os.path.join(
        #             single_experiment['pathname_experience'],
        #             '{name}'.format(name=name))
        #         arguments['output_path'] = output_path
        #         arguments['feature_path'] = test_features
        #         arguments['network_path'] = model.output_path + '.pth'
        #         embedder = embedder_class(**arguments)
        #         print("Embedding test features {} at path {}"
        #               .format(name, output_path))
        #         embedder.embed()

    def run(self):
        """Run command to launch the grid search

        """
        self.parse_yaml_input_file()

        msg_yaml_error = 'Yaml not well formatted : '
        assert self.params['default_params'], msg_yaml_error + 'default_params'
        default_params = self.params['default_params']

        # fill test files in the yaml
        if "test_files" in self.params:
            for test_file in self.params["test_files"]:
                self.test_files.append(self.params["test_files"][test_file])

        self.run_embedding(single_experiment=default_params)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_yml", type=str,
                           help="yaml file for the experiment")
    argparser.add_argument("-w", "--weights", type=str,
                           help="Path to weights")
    argparser.add_argument("-i", "--input-features", type=str,
                           help="Path to input features. Default will be those"
                                "in the yaml file")

    argparser.add_argument("-o", "--output")

    args = argparser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    t1 = time.time()
    print("Start embedding")
    grid = EmbedCLI(yaml_file=args.exp_yml,
                    weights=args.weights,
                    input_features=args.input_features
                    )

    grid.run()
    print("The embedding took {} s ".format(time.time() - t1))


if __name__ == '__main__':
    main()
