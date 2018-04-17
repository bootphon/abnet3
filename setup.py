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
import subprocess


class RequiredPackagedInstall(install):
    """ Install needed Github repositories"""

    def run(self):
        """ Get DTW_Cython, H5features & Spectral
        from github.com/bootphon, by launching pip via
        subprocess
        """
        # run ABnet3 package installation
        install.run(self)


with io.open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    name='abnet3',
    version='0.0.1',
    packages=['abnet3'],
    description='ABnet neural network in Pytorch',
    author='Rachid Riad',
    license='GPLv3',
    cmdclass={
        'install': RequiredPackagedInstall,
    },
    entry_points={'console_scripts': [
        'abnet3-gridsearch = abnet3.gridsearch:main',
    ]}
)
