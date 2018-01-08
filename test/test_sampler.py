#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pytest
from abnet3.sampler import SamplerClusterSiamese

class TestSampler:

    def test_parse_input_file(self):
        base_path = os.path.dirname(__file__)
        file_name = os.path.join(base_path, "data/english.test.classes")
        sampler = SamplerClusterSiamese()

        # test without max clusters
        clusters = sampler.parse_input_file(input_file=file_name)
        assert clusters == [[['s0102a', 149.359, 149.66]],
                            [['s2401a', 70.782, 71.282],
                             ['s2402b', 14.639, 15.234],
                             ['s2403b', 96.311, 96.739],
                             ['s2404b', 96.311, 96.739],
                             ['s2405b', 96.311, 96.739],
                             ],
                            [['s2403a', 258.748, 259.267]],
                            [['s0102a', 152.623, 153.083]],
                            [['s2702a', 31.902, 32.37]],
                            [['s0101a', 295.416, 295.955], ['s0101a', 546.471, 546.681]],
                            [['s2001a', 217.712, 218.591], ['s2001a', 546.471, 546.681]]]

        # test with max clusters
        clusters = sampler.parse_input_file(file_name, max_num_clusters=3)
        assert len(clusters) == 3

    def test_split_cluster_ratio(self):
        base_path = os.path.dirname(__file__)
        file_name = os.path.join(base_path, "data/english.test.classes")
        sampler = SamplerClusterSiamese()

        clusters = sampler.parse_input_file(file_name)

        train_clusters, dev_clusters = sampler.split_clusters_ratio(clusters)

        # check they have the same number of words
        n_words_train = sum([len(cluster) for cluster in train_clusters])
        n_words_dev = sum([len(cluster) for cluster in dev_clusters])
        n_words = sum([len(cluster) for cluster in clusters])
        assert n_words_train + n_words_dev == n_words

        # same check with max_cluster size
        sampler = SamplerClusterSiamese(max_size_cluster=3)
        train_clusters, dev_clusters = sampler.split_clusters_ratio(clusters)
        n_words_train = sum([len(cluster) for cluster in train_clusters])
        n_words_dev = sum([len(cluster) for cluster in dev_clusters])
        n_words = sum([len(cluster) for cluster in clusters])
        assert n_words_train + n_words_dev == n_words

        # check that no cluster is bigger than 3
        assert max([len(cluster) for cluster in train_clusters]) <= 3
