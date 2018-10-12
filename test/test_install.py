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
    try: # for pip >= 10
        from pip._internal.req import parse_requirements
    except ImportError: # for pip <= 9.0.3
        from pip.req import parse_requirements

    try:
        session = pip.download.PipSession()
    except AttributeError:  # for pip >= 10
        session = pip._internal.download.PipSession()

    requirements_path = pathlib.Path(__file__).parents[1].joinpath(
                        'requirements.txt')
    requirements = parse_requirements(str(requirements_path), session=session)
    requirements = [str(r.req) for r in requirements]
    requirements = [r for r in requirements if r != 'None']
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
