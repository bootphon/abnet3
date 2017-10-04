#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''abnet3: Siamese Neural Network for speech
Note that "python setup.py test" invokes pytest on the package. With
appropriately configured setup.cfg, this will check both xxx_test
modules and docstrings.
Copyright 2017, Rachid Riad, LSCP
Licensed under GPLv3.
'''

import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


# This is a plug-in for setuptools that will invoke py.test
# when you run python setup.py test
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, because outside the required eggs aren't loaded
        # yet
        import pytest
        sys.exit(pytest.main(self.test_args))


version = "0.1"

setup(name="abnet3",
      version=version,
      description="Siamese Neural Network for speech in Pytorch",
      long_description=open("README.rst").read(),
      classifiers=[
          # Get strings from
          # http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 1 - Planning',
          'Programming Language :: Python3'],
      keywords="Speech",  # Separate with spaces
      author="Rachid Riad",
      author_email="riadrachid3@gmail.com",
      url="",
      license="GPLv3",
      packages=find_packages(exclude=['examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass={'test': PyTest},

      # TODO: List of packages that this one depends upon:
      dependency_links=[
          'https://github.com/bootphon/spectral/tarball/master'
          '#egg=spectral'],
      install_requires=[
          'cython', 'numpy','scipy', 'pytorch', 'spectral'],

      # TODO: List executable scripts, provided by the package (this
      # is just an example)
      entry_points={
        'console_scripts': ['abnet3=abnet3:main']
      }
)