#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:24:47 2017

@author: Rachid Riad
"""

import pytest
import pathlib
import unittest
import pip
import pkg_resources


def test_requirements():  # pylint: disable=no-self-use
    """Recursively confirm that requirements are available.

    This implementation is tested to be compatible with pip 9.0.1.
    Test stolen from stack-overflow
    """
    requirements_path = pathlib.Path(__file__).parents[1].joinpath(
                        'requirements.txt')
    requirements = pip.req.parse_requirements(
                   str(requirements_path), session=pip.download.PipSession())
    requirements = [str(r.req) for r in requirements]
    pkg_resources.require(requirements)


def test_pytorch():
    """Test the pytorch installation

    """
    dependencies = [
      'torch>=0.2'
          ]
    pkg_resources.require(dependencies)


if __name__ == '__main__':
    pytest.main([__file__])
