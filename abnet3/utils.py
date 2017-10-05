#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  3 19:22:41 2017

@author: Rachid Riad
"""

import numpy as np
import os


def get_var_name(**variable):
    return list(variable.keys())[0]


def normalize_distribution(p):
    """Normalize distribution p for a dictionnary k/v class
    
    """
    assert type(p)==dict, 'Distribution p is not a dictionnary'
    sum_norm = 0.0
    keys = p.keys()
    for key in keys:
        sum_norm += p[key]
    
    for key in keys:
        p[key] = p[key]/sum_norm

    return p
    

# Inspiration from numpy code source 

def cumulative_distribution(distribution):
    """Cumulative sums for multinomial sampling
    
    """
    assert type(distribution) == list, 'distribution variable needs to be list'
    assert np.sum(distribution) == 1.0, 'distribution needs to be normalized'
    
    cdf = np.cumsum(np.array(distribution))
    cdf /= cdf[-1]
    return cdf


def sample_searchidx(cdf, num_samples):
    """Sample indexes based on cdf distribution
    
    """
    uniform_samples = np.random.random_sample(num_samples)
    idx = cdf.searchsorted(uniform_samples, side='right')
    return idx

def print_token(tok):
    """Pretty print token for batches
    
    """
    return "{0} {1:.2f} {2:.2f}".format(tok[0], tok[1], tok[2])


def Parse_Dataset(path):
    """Parse folder for batch names
    
    """
    batches = []
    batches += ([os.path.join(path, add) for add in os.listdir(path) if add.endswith(('.batch'))])
    return batches






