#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script creates hdf5 feature files from the wav files.
"""

from spectral import Spectral
import numpy as np
import h5features
from scipy.io import wavfile
import os
import h5py
import shutil
import tempfile


def do_fbank(fname):
    """Compute standard filterbanks from a wav file"""
    srate, sound = wavfile.read(fname)
    fbanks = Spectral(
        nfilt=40,               # nb of filters in mel bank
        alpha=0.97,             # pre-emphasis
        do_dct=False,           # we do not want MFCCs
        fs=srate,               # sampling rate
        frate=100,              # frame rate
        wlen=0.025,             # window length
        nfft=1024,              # length of dft
        do_deltas=False,        # speed
        do_deltasdeltas=False   # acceleration
    )
    fb = np.array(fbanks.transform(sound), dtype='float32')
    return fb


def do_mfccs(fname):
    """Compute standard mfccs from a wav file"""
    srate, sound = wavfile.read(fname)
    fbanks = Spectral(
        nfilt=40,               # nb of filters in mel bank
        alpha=0.97,             # pre-emphasis
        fs=srate,               # sampling rate
        frate=100,              # frame rate
        wlen=0.025,             # window length
        nfft=512,               # length of dft
        ncep=13,                # nb of cepstral coefficients
        lowerf=100,
        upperf=6855.4976,
        do_deltas=False,        # speed
        do_deltasdeltas=False   # acceleration
    )
    fb = np.array(fbanks.transform(sound), dtype='float32')
    return fb


def stack_fbanks(features, nframes=7):
    """Stack input features. Each frame is now the concatenation of its
    previous frames and its next frames.

    First and last frames are padded with zeros.

    Parameters:
    ----------
    features: array, input features
    nframes: int, number of frames to stack.

    Returns:
    -------
    features_s: array, stacked fbanks.
        (shape = (fbanks.shape[0], fbanks.shape[1]*nframes)).
    """
    assert nframes % 2 == 1, 'number of stacked frames must be odd'
    dim = features.shape[1]
    pad = np.zeros((nframes/2, dim), dtype=features.dtype)
    features = np.concatenate((pad, features, pad))
    aux = np.array([features[i:i-nframes+1]
                    for i in range(nframes-1)] + [features[nframes-1:]])
    return np.reshape(np.swapaxes(aux, 0, 1), (-1, dim * nframes))


def(files, h5f, featfunc=do_fbank, timefunc=None):
    """Compute mfcc or filterbanks (or other) in h5features format.

    Parameters:
    ----------
    files: list, list of files on which to compute the features. You must
        give the complete relative or absolute path of the wave file
    h5f: str. Name of the h5features file to create.
    featfunc: callable. "do_fbanks" to compute fbanks, "do_mfccs" to compute
        mfccs. Or any callable function that return features given a wave file.
    timefunc: callable. Function that returns timestamps for the aforementionned
        features. By default, it assume a window length of 25 ms and a window
        step of 10 ms.
    """
    batch_size = 500
    features = []
    times = []
    internal_files = []
    i = 0
    for f in files:
        if i == batch_size:
            h5features.write(h5f, '/features/', internal_files, times,
                             features)
            features = []
            times = []
            internal_files = []
            i = 0
        i = i+1
        data = featfunc(f)
        features.append(data)
        if timefunc == None:
            time = np.arange(data.shape[0], dtype=float) * 0.01 + 0.0025
        else:
            time = timefunc(f)
        times.append(time)
        internal_files.append(os.path.basename(os.path.splitext(f)[0]))
    if features:
        h5features.write(h5f, '/features/',
                         internal_files, times,
                         features)


def mean_variance_normalisation(h5f, mvn_h5f, vad=None):
    """Do mean variance normlization. Optionnaly use a vad.

    Parameters:
    ----------
    h5f: str. h5features file name
    mvn_h5f: str, h5features output name
    """
    dset = h5py.File(h5f).keys()[0]
    if vad is not None:
        raise NotImplementedError
    else:
        data = h5py.File(h5f)[dset]['features'][:]
        features = data
    epsilon = np.finfo(data.dtype).eps
    mean = np.mean(data)
    std = np.std(data)
    mvn_features = (features - mean) / (std + epsilon)
    shutil.copy(h5f, mvn_h5f)
    h5py.File(mvn_h5f)[dset]['features'][:] = mvn_features


def h5features_feats2stackedfeats(fb_h5f, stackedfb_h5f, nframes=7):
    """Create stacked features version of h5features file

    Parameters:
    ----------
    fb_h5f: str. h5features file name
    stackedfb_h5f: str, h5features output name
    """
    dset_name = h5py.File(fb_h5f).keys()[0]
    files = h5py.File(fb_h5f)[dset_name]['items']
    def aux(f):
        return stack_fbanks(h5features.read(fb_h5f, from_item=f)[1][f],
                            nframes=nframes)
    def time_f(f):
        return h5features.read(fb_h5f, from_item=f)[0][f]
    h5features_compute(files, stackedfb_h5f, featfunc=aux,
                      timefunc=time_f)


def generate_all(files, alignement_h5f, input_h5f,
                 nframes=7, vad=None):
    """Generate mfccs and stacked mean variance normalized filterbanks
    for the input files.

    Parameters:
    ----------
    files: list, list of files on which to compute the features. You must
        give the complete relative or absolute path of the wave file
    alignement_h5f: str. Name of the h5features file containing the alignment
        features. Features which will be used to align "same" word for the
        abnet, here mfccs.
    input_h5f: str. Name of the h5features file containing the input features
        for the abnet. Here stacked mean variance normalized filterbanks.
    nframes: int, number of frames to stack.
    """
    def try_remove(fname):
        try:
            os.remove(fname)
        except:
            pass
    try:
        directory = os.path.dirname(os.path.abspath(input_h5f))

        # create temporary files:
        _, fb_h5f = tempfile.mkstemp(dir=directory)
        _, fb_mvn_h5f = tempfile.mkstemp(dir=directory)
        os.remove(fb_h5f)
        os.remove(fb_mvn_h5f)

        # generate mfccs:
        h5features_compute(files, alignement_h5f, featfunc=do_mfccs)

        # generate stacked mvn fbanks:
        h5features_compute(files, fb_h5f, featfunc=do_fbank)
        mean_variance_normalisation(fb_h5f, fb_mvn_h5f, vad=vad)
        h5features_feats2stackedfeats(fb_mvn_h5f, input_h5f, nframes=nframes)
    finally:
        try_remove(fb_h5f)
        try_remove(fb_mvn_h5f)