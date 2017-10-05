#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''abnet3: Siamese Neural Network for speech
Note that "python setup.py test" invokes pytest on the package. With
appropriately configured setup.cfg, this will check both xxx_test
modules and docstrings.
Copyright 2017, Rachid Riad, LSCP
Licensed under GPLv3.
'''

from setuptools import setup, find_packages
import sys
import io

sys.path.append("./abnet3")

with io.open("requirements.txt", encoding="utf-8") as req_fp:
  install_requires = req_fp.readlines()

setup(
  name='abnet3',
  version='0.0.1',
  description='ABnet neural network in Pytorch',
  author='Rachid Riad',
  license='GPLv3',
  install_requires=install_requires,
  packages=[
      'abnet3',
  ],
)
