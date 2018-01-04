#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pytest
from abnet3.sampler import SamplerClusterSiamese


def test_parse_input_file():
    base_path = os.path.dirname(__file__)

    file_name = os.path.join(base_path, "data/english.test.classes")
    sampler = SamplerClusterSiamese()

    # test without max clusters
    clusters = sampler.parse_input_file(input_file=file_name)
    assert clusters == [[['s0102a', 149.359, 149.66]],
                        [['s2401a', 70.782, 71.282], ['s2402b', 14.639, 15.234], ['s2402b', 96.311, 96.739]],
                        [['s2403a', 258.748, 259.267]],
                        [['s0102a', 152.623, 153.083]],
                        [['s2702a', 31.902, 32.37]],
                        [['s0101a', 295.416, 295.955], ['s0101a', 546.471, 546.681]],
                        [['s2001a', 217.712, 218.591], ['s2001a', 546.471, 546.681]]]


    # test with max clusters
    clusters = sampler.parse_input_file(file_name, max_num_clusters=3)
    print(clusters)
    assert len(clusters) == 3