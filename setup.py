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
from setuptools.command.install import install

import sys
import io


class RequiredPackagedInstall(install):
    """ Install needed Github repositories"""

    def run(self):
        """ Get DTW_Cython, H5features & Spectral 
        from github.com/bootphon, by launching pip via
        subprocess
        """
        import pip

        # github repositories we want to get
        DTW_cython = "git+http://github.com/Rachine/DTW_Cython.git"
        spectral = "git+http://github.com/Rachine/DTW_Cython.git"
        h5features = "git+https://github.com/bootphon/h5features.git"

        # install with pip
        pip.main(['install', DTW_cython])
        pip.main(['install', spectral])
        pip.main(['install', h5features])


sys.path.append("./abnet3")

with io.open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
  name='abnet3',
  version='0.0.1',
  description='ABnet neural network in Pytorch',
  author='Rachid Riad',
  license='GPLv3',
  packages=[
      'abnet3',
  ],
  cmdclass={
      'install': RequiredPackagedInstall,
  }
)
