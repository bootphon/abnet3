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


class FeaturesGenerator:

    def __init__(self, n_filters=40, method='fbanks', normalization=True, stack=True,
                 nframes=7, deltas=False, deltasdeltas=False):
        self.n_filters = n_filters
        self.method = method
        self.normalization = normalization
        self.stack = stack
        self.nframes = nframes
        self.deltas = deltas
        self.deltasdeltas = deltasdeltas

        if self.method not in ['mfcc', 'fbanks']:
            raise ValueError("Method %s not recognized" % self.method)

    def do_fbank(self, fname):
        """Compute standard filterbanks from a wav file"""
        srate, sound = wavfile.read(fname)
        fbanks = Spectral(
            nfilt=self.n_filters,   # nb of filters in mel bank
            alpha=0.97,             # pre-emphasis
            do_dct=False,           # we do not want MFCCs
            fs=srate,               # sampling rate
            frate=100,              # frame rate
            wlen=0.025,             # window length
            nfft=1024,              # length of dft
            do_deltas=self.deltas,        # speed
            do_deltasdeltas=self.deltasdeltas   # acceleration
        )
        fb = np.array(fbanks.transform(sound), dtype='float32')
        return fb

    def do_mfccs(self, fname):
        """Compute standard mfccs from a wav file"""
        srate, sound = wavfile.read(fname)
        fbanks = Spectral(
            nfilt=self.n_filters,   # nb of filters in mel bank
            alpha=0.97,             # pre-emphasis
            fs=srate,               # sampling rate
            frate=100,              # frame rate
            wlen=0.025,             # window length
            nfft=512,               # length of dft
            ncep=13,                # nb of cepstral coefficients
            lowerf=100,
            upperf=6855.4976,
            do_deltas=self.deltas,        # speed
            do_deltasdeltas=self.deltasdeltas   # acceleration
        )
        fb = np.array(fbanks.transform(sound), dtype='float32')
        return fb

    def stack_fbanks(self, features, nframes=7):
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
        pad = np.zeros((nframes//2, dim), dtype=features.dtype)
        features = np.concatenate((pad, features, pad))
        aux = np.array([features[i:i-nframes+1]
                        for i in range(nframes-1)] + [features[nframes-1:]])
        return np.reshape(np.swapaxes(aux, 0, 1), (-1, dim * nframes))

    def h5features_compute(self, files, h5f, featfunc=None, timefunc=None):
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
        if featfunc is None:
            featfunc = self.do_fbank
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

    def mean_variance_normalisation(self, h5f, mvn_h5f, vad=None):
        """Do mean variance normlization. Optionnaly use a vad.

        Parameters:
        ----------
        h5f: str. h5features file name
        mvn_h5f: str, h5features output name
        """

        dset = list(h5py.File(h5f).keys())[0]

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

    def h5features_feats2stackedfeats(self, fb_h5f, stackedfb_h5f, nframes=7):
        """Create stacked features version of h5features file

        Parameters:
        ----------
        fb_h5f: str. h5features file name
        stackedfb_h5f: str, h5features output name
        """
        dset_name = list(h5py.File(fb_h5f).keys())[0]
        files = h5py.File(fb_h5f)[dset_name]['items']

        def aux(f):
            return self.stack_fbanks(
                h5features.read(fb_h5f, from_item=f)[1][f],
                nframes=nframes)

        def time_f(f):
            return h5features.read(fb_h5f, from_item=f)[0][f]
        self.h5features_compute(
            files, stackedfb_h5f, featfunc=aux,
            timefunc=time_f)

    def generate(self, files, output_path):
        """
        :param list files: List of wav files.  You must
            give the complete relative or absolute path of the wave file
        :param str output_path: path where the features will be saved
        :param str method: can be either 'mfcc' or 'fbank'
        :param bool stack: stack features with block of `nframes` features.
        :param bool normalization: mean / variance normalization
        :param int nframes: number of frames to stack (if stack is True)
        """

        functions = {
            'mfcc': self.do_mfccs,
            'fbank': self.do_fbank
        }

        if self.method not in functions:
            raise ValueError("Method %s not authorized." % self.method)
        f = functions[self.method]

        tempdir = tempfile.mkdtemp()
        h5_temp1 = tempdir + '/temp1'
        print("Spectral transforming with %s" % self.method)
        self.h5features_compute(files, h5_temp1, featfunc=f)

        if self.normalization:
            print("Normalizing")
            h5_temp2 = tempdir + '/temp2'
            self.mean_variance_normalisation(h5_temp1, h5_temp2)
        else:
            h5_temp2 = h5_temp1
        if self.stack:
            print("Stacking frames")
            self.h5features_feats2stackedfeats(h5_temp2, output_path, nframes=self.nframes)
        else:
            shutil.copy(h5_temp2, output_path)

        shutil.rmtree(tempdir)
