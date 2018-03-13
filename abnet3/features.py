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

from abnet3.utils import read_vad_file, read_feats, Features_Accessor


class FeaturesGenerator:

    def __init__(self, files=None, output_path=None,
                 load_mean_variance_path=None,
                 save_mean_variance_path=None,
                 vad_file=None,
                 n_filters=40, method='fbanks', normalization=True,
                 norm_per_file=False, stack=True,
                 nframes=7, deltas=False, deltasdeltas=False,
                 norm_per_channel=True,
                 run='once'):
        """

        :param files: list of wav file paths, or folder
        :param output_path: Location of the output h5features file
        :param load_mean_variance_path: (optional)
            Should be None, or the path to an existing file.
            If this is a path, it will use the file at the provided location
            to inject the mean and variance into the wav files.
            It will not compute true mean and variance
        :param save_mean_variance_path: (optional)
            Should be None, or the path to an existing file.
            If this is a path, it will
            save the mean and variance of the dataset at the given path.
            It can then be used with load_mean_variance_path for another
            dataset.
        :param vad_file: (optional)
            Path to a file with VAD data. If given, the mean and variance
            of the dataset will be computed only in non-silent regions
        :param n_filters: number of filters in spectral
        :param method: mfcc or fbanks
        :param normalization: if True, normalize the dataset
        :param norm_per_file: if True, normalize within each wav file.
        :param stack: if True, will stack the data (stacks of nframes
            consecutive frames)
        :param nframes: number of frames in a stack (if stack is True)
        :param deltas: first order derivative
        :param deltasdeltas: second order derivative
        :param run: Param to notify if features has to be computed
        """

        self.files = files
        self.output_path = output_path
        self.load_mean_variance_path = load_mean_variance_path
        self.save_mean_variance_path = save_mean_variance_path
        self.vad_file = vad_file
        self.n_filters = n_filters
        self.method = method
        self.normalization = normalization
        self.stack = stack
        self.nframes = nframes
        self.deltas = deltas
        self.deltasdeltas = deltasdeltas
        self.norm_per_file = norm_per_file
        self.norm_per_channel = norm_per_channel
        self.run = run

        if self.method not in ['mfcc', 'fbanks']:
            raise ValueError("Method %s not recognized" % self.method)
        assert self.run in ['never', 'once', 'always']

        if load_mean_variance_path is not None \
                and save_mean_variance_path is not None:
            raise ValueError("You can't both read and save mean and variance")
        if not self.normalization and self.norm_per_file:
            raise ValueError("You can't set normalization to False "
                             "and normalization per file to True.")

        if self.norm_per_file and (load_mean_variance_path is not None or
                                   save_mean_variance_path is not None):
            raise ValueError("You can't compute mean and variance "
                             "per file and loading / saving it.")

    def whoami(self):
        return {
            'params': self.__dict__,
            'class_name': self.__class__.__name__,
        }

    def do_fbank(self, fname):
        """Compute standard filterbanks from a wav file"""
        srate, sound = wavfile.read(fname)
        fbanks = Spectral(
            nfilt=self.n_filters,  # nb of filters in mel bank
            alpha=0.97,  # pre-emphasis
            do_dct=False,  # we do not want MFCCs
            fs=srate,  # sampling rate
            frate=100,  # frame rate
            wlen=0.025,  # window length
            nfft=1024,  # length of dft
            do_deltas=self.deltas,  # speed
            do_deltasdeltas=self.deltasdeltas  # acceleration
        )
        fb = np.array(fbanks.transform(sound), dtype='float32')
        return fb

    def do_mfccs(self, fname):
        """Compute standard mfccs from a wav file"""
        srate, sound = wavfile.read(fname)
        fbanks = Spectral(
            nfilt=self.n_filters,  # nb of filters in mel bank
            alpha=0.97,  # pre-emphasis
            fs=srate,  # sampling rate
            frate=100,  # frame rate
            wlen=0.025,  # window length
            nfft=512,  # length of dft
            ncep=13,  # nb of cepstral coefficients
            lowerf=100,
            upperf=6855.4976,
            do_deltas=self.deltas,  # speed
            do_deltasdeltas=self.deltasdeltas  # acceleration
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
        pad = np.zeros((nframes // 2, dim), dtype=features.dtype)
        features = np.concatenate((pad, features, pad))
        aux = np.array(
            [features[i:i - nframes + 1] for i in range(nframes - 1)] +
            [features[nframes - 1:]]
        )
        return np.reshape(np.swapaxes(aux, 0, 1), (-1, dim * nframes))

    def h5features_compute(self, files, h5f, featfunc=None, timefunc=None):
        """Compute mfcc or filterbanks (or other) in h5features format.

        Parameters:
        ----------
        files: list, list of files on which to compute the features. You must
            give the complete relative or absolute path of the wave file
        h5f: str. Name of the h5features file to create.
        featfunc: callable. "do_fbanks" to compute fbanks, "do_mfccs" to
            compute mfccs. Or any callable function that return features
            given a wave file.
        timefunc: callable. Function that returns timestamps for the
            aforementionned features. By default, it assume a window length
            of 25 ms and a window step of 10 ms.
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
            i = i + 1
            data = featfunc(f)
            features.append(data)
            if timefunc is None:
                time = np.arange(data.shape[0], dtype=float) * 0.01 + 0.0025
            else:
                time = timefunc(f)
            times.append(time)
            internal_files.append(os.path.basename(os.path.splitext(f)[0]))
        if features:
            h5features.write(h5f, '/features/',
                             internal_files, times,
                             features)

    def mean_variance_normalisation(self, h5f, mvn_h5f, params=None,
                                    vad_file=None):
        """Do mean variance normlization. Optionnaly use a vad.

        Parameters:
        ----------
        h5f: str. h5features file name
        mvn_h5f: str, h5features output name
        params : dict {'mean': mean, 'variance': variance}
        """
        # normalize either per channel or on the whole spectrum.
        axis = 0 if self.norm_per_channel else None

        features_accessor, _, _ = read_feats(h5f)

        # VAD
        if vad_file is not None:
            vad_data = read_vad_file(vad_file)
            self.filter_vad_whole_dataset(features_accessor, vad_data)

        features = np.vstack(features_accessor.features.values())
        epsilon = np.finfo(features.dtype).eps

        # calculate mean
        if params is not None:
            mean = params['mean']
            std = params['variance']
        else:
            mean = np.mean(features, axis=axis)
            std = np.std(features, axis=axis)
        del features, features_accessor  # free memory

        # we reload all the features because we wan't to keep them in the
        # right order to put them back in the h5py file.
        dset = list(h5py.File(h5f).keys())[0]
        features = h5py.File(h5f)[dset]['features'][:]
        mvn_features = (features - mean) / (std + epsilon)
        shutil.copy(h5f, mvn_h5f)
        h5py.File(mvn_h5f)[dset]['features'][:] = mvn_features
        return mean, std

    def filter_vad_whole_dataset(self, features_accessor,  vad_data):
        """
        This function will filter the feature
        :param features_accessor: feature accessor
        :param vad_data:
            dictionnary {'file': [[s, e], [s, e], ...]
        :return:
        """
        for file in vad_data:
            filtered_features = []
            for start, end in vad_data[file]:
                filtered_features.append(features_accessor.get(file,
                                                               start, end))
            features_accessor.features[file] = np.concatenate(filtered_features)

    def filter_vad_one_file(self, feature, time, vad_data):
        filtered_features = []
        for start, end in vad_data:
            filtered_features.append(Features_Accessor.get_features_between(
                feature, time, start, end))
        return np.concatenate(filtered_features)

    def mean_var_norm_per_file(self, h5f, mvn_h5f, vad_file=None):
        # normalize either per channel or on the whole spectrum.
        axis = 0 if self.norm_per_channel else None

        dset_name = list(h5py.File(h5f).keys())[0]
        files = h5py.File(h5f)[dset_name]['items']
        reader = h5features.Reader(h5f)
        means_vars = []
        for f in files:
            data = reader.read(from_item=f)
            items, features, times = (data.items(), data.features()[0],
                                      data.labels()[0])
            # VAD
            filtered_features = None
            if vad_file is not None:
                vad_data = read_vad_file(vad_file)
                if str(f) in vad_data:
                    filtered_features = self.filter_vad_one_file(
                        features, times, vad_data[str(f)])

            if filtered_features is None:
                mean = np.mean(features, axis=axis)
                std = np.std(features, axis=axis)
            else:
                mean = np.mean(filtered_features, axis=axis)
                std = np.std(filtered_features, axis=axis)
            features = (features - mean) / (std + np.finfo(features.dtype).eps)
            h5features.write(mvn_h5f, '/features/', items, [times], [features])
            means_vars.append((f, mean, std))
        return means_vars

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

    def save_mean_variance(self, mean, variance, output_file):
        """
        This function will save the mean and variance into a folder.
        It will take the form

        mean variance (file: optional)

        :param mean: np.array
        :param variance: np.array
        :param output_file: file where mean and variance will be saved
        """
        mean_var = np.vstack((mean, variance))
        np.savetxt(output_file, mean_var)

    def load_mean_variance(self, file_path):
        """
        :return: a dict {'mean': mean, 'variance': variance}
        """
        mean_var = np.loadtxt(file_path)

        return {'mean': mean_var[0], 'variance': mean_var[1]}

    def generate(self):

        functions = {
            'mfcc': self.do_mfccs,
            'fbanks': self.do_fbank
        }

        if type(self.files) == str:
            if not os.path.isdir(self.files):
                raise ValueError("files must be a directory or a list of "
                                 "files")
            self.files = [os.path.join(self.files, f)
                          for f in os.listdir(self.files)
                          if f.endswith('.wav')]

        if self.method not in functions:
            raise ValueError("Method %s not authorized." % self.method)
        f = functions[self.method]

        tempdir = os.path.join(os.path.dirname(self.output_path), 'tmp')
        os.makedirs(tempdir, exist_ok=True)
        h5_temp1 = tempdir + '/temp1'
        print("Spectral transforming with %s" % self.method)
        self.h5features_compute(self.files, h5_temp1, featfunc=f)

        if self.normalization:
            print("Normalizing")
            h5_temp2 = tempdir + '/temp2'
            if self.norm_per_file:
                self.mean_var_norm_per_file(h5_temp1, h5_temp2,
                                            vad_file=self.vad_file)
            else:
                if self.load_mean_variance_path is not None:
                    params = self.load_mean_variance(
                        file_path=self.load_mean_variance_path)
                else:
                    params = None
                mean, variance = self.mean_variance_normalisation(
                    h5_temp1, h5_temp2, params=params,
                    vad_file=self.vad_file
                )
                if self.save_mean_variance_path is not None:
                    self.save_mean_variance(
                        mean, variance,
                        output_file=self.save_mean_variance_path)
        else:
            h5_temp2 = h5_temp1
        if self.stack:
            print("Stacking frames")
            self.h5features_feats2stackedfeats(h5_temp2, self.output_path,
                                               nframes=self.nframes)
        else:
            shutil.copy(h5_temp2, self.output_path)

        shutil.rmtree(tempdir)
