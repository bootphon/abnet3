#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script is composed of the different modules for sampling pairs
based on gold words or spoken term discovery outputs.

It will generate text files in two folders 'train_pairs' and 'dev_pairs' in the
correct format for the Neural Network.

"""

from abnet3.utils import normalize_distribution, cumulative_distribution
from abnet3.utils import print_token, sample_searchidx
from abnet3.utils import read_spkid_file, read_spk_list, progress

import numpy as np
import os
import codecs
import random
from collections import defaultdict


class SamplerBuilder(object):
    """Sampler Model interface

        Parameters
        ----------
        batch_size : Int
            Number of words per batch
        run : String
            Param to notify if features has to be computed
        directory_output : String
            Path folder where train/dev pairs folder will be
        seed : int
            Seed

    """
    def __init__(self, batch_size=8, run='once', input_file=None,
                 directory_output=None, ratio_train_dev=0.7, seed=0):
        super(SamplerBuilder, self).__init__()
        self.batch_size = batch_size
        self.run = run
        self.directory_output = directory_output
        self.seed = seed
        self.ratio_train_dev = ratio_train_dev
        assert self.run in ['never', 'once', 'always']

    def whoami(self):
        raise NotImplementedError('Unimplemented whoami for class:',
                                  self.__class__.__name__)

    def parse_input_file(self, input_file=None):
        """Parse input file to prepare sampler

        """
        raise NotImplementedError('Unimplemented parse_input_file for class:',
                                  self.__class__.__name__)

    def sample_batch(self):
        """Generic function to generate a batch for network

        """
        raise NotImplementedError('Unimplemented sample_batch for class:',
                                  self.__class__.__name__)


class SamplerPairs(SamplerBuilder):
    """Sampler Model interface based on pairs of similar words

    """
    def __init__(self, *args, **kwargs):
        super(SamplerPairs, self).__init__(*args, **kwargs)


class SamplerCluster(SamplerBuilder):
    """Sampler Model interface based on clusters of words

    Parameters
    ----------
    std_file : String
        Path to the file with the cluster of words
    spk_list_file : String
        Path to the file with the speaker list
    spkid_file : String
        Path to the file with file_id to Speaker_id mapping
    type_sampling_mode : String
        function applied to the observed type frequencies
    spk_sampling_mode : String
        function applied to the observed speaker frequencies
    create_batches: bool
        If you want the sampler to save one file for each dataset,
        or multiple batches

    """
    def __init__(self, max_size_cluster=10, ratio_same_diff_spk=0.75,
                 ratio_same_diff_type=0.5,
                 type_sampling_mode='log', spk_sampling_mode='log',
                 std_file=None, spk_list_file=None, spkid_file=None,
                 max_num_clusters=None,
                 sample_batches=False,
                 num_total_sampled_pairs=None,
                 *args, **kwargs):
        super(SamplerCluster, self).__init__(*args, **kwargs)
        self.max_size_cluster = max_size_cluster
        self.ratio_same_diff_spk = ratio_same_diff_spk
        self.ratio_same_diff_type = ratio_same_diff_type
        self.type_sampling_mode = type_sampling_mode
        self.spk_sampling_mode = spk_sampling_mode
        self.std_file = std_file
        self.spk_list_file = spk_list_file
        self.spkid_file = spkid_file
        self.max_num_clusters = max_num_clusters
        self.sample_batches = sample_batches
        self.num_total_sampled_pairs = num_total_sampled_pairs

    def parse_input_file(self, input_file=None, max_num_clusters=None):
        """Parse input file:

        Parameters
        ----------
        input_file : String
            Path to clusters of words
        max_num_clusters : int
            Number max of clusters, useful for debugging
        """
        print("parsing input file")
        with codecs.open(input_file, "r", "utf-8") as fh:
            lines = fh.readlines()
        clusters = []
        i = 0
        while i < len(lines):
            cluster = []
            tokens = lines[i].strip().split(" ")
            assert len(tokens) == 2, 'problem line {} '.format(i) + str(tokens)
            i = i+1
            tokens = lines[i].strip().split(" ")
            assert len(tokens) == 3, "Empty class!"
            fid, t0, t1 = tokens
            t0, t1 = float(t0), float(t1)
            cluster.append([fid, t0, t1])
            new_class = False
            while not(new_class):
                i = i+1
                tokens = lines[i].strip().split(" ")
                if len(tokens) == 3:
                    fid, t0, t1 = tokens
                    t0, t1 = float(t0), float(t1)
                    cluster.append([fid, t0, t1])
                else:
                    assert tokens == ['']
                    new_class = True
                    clusters.append(cluster)
                    i = i+1

        # select the clusters we will keep
        if max_num_clusters is not None and \
                0 < max_num_clusters < len(clusters):
            clusters = random.sample(clusters, max_num_clusters)
        return clusters

    def split_clusters_ratio(self, clusters):
        """Split clusters, two type of splits involved for the train and
            dev. Biggest clusters are going to be split with the ratio and
            the result parts goes to train and dev. The other smaller
            clusters are then chosen randomly to go either in the train or dev.

            Parameters
            ----------
            clusters : list
                List of cluster parsed from parse_input_file
            ratio_train_dev: float
                Ratio number between 0 and 1 to randomly split train and
                dev clusters
        """

        train_clusters, dev_clusters = [], []
        num_clusters = len(clusters)
        num_train = int(self.ratio_train_dev*num_clusters)
        train_idx = np.random.choice(num_clusters, num_train, replace=False)

        for idx, cluster in enumerate(clusters):
            # Tricky move here to split big clusters for train and dev
            size_cluster = len(cluster)
            if self.max_size_cluster > 1 and \
               self.max_size_cluster < size_cluster:
                num_train = int(self.ratio_train_dev*size_cluster)
                indexes = range(size_cluster)
                rand_idx = np.random.permutation(indexes)
                train_split = [cluster[spec_idx] for spec_idx
                               in rand_idx[:num_train]]
                dev_split = [cluster[spec_idx] for spec_idx
                             in rand_idx[num_train:]]
                train_clusters.append(train_split)
                dev_clusters.append(dev_split)
            else:
                if idx in train_idx:
                    train_clusters.append(cluster)
                else:
                    dev_clusters.append(cluster)

        return train_clusters, dev_clusters

    def analyze_clusters(self, clusters, get_spkid_from_fid=None):
        """Analysis input file to prepare sampler

        Parameters
        ----------
        clusters : list
            List of cluster parsed from parse_input_file
        get_spkid_from_fid : dict
            Mapping dictionnary between files and speaker id


        Returns : {
            tokens : the list of tokens
            token_types : cluster id for each token
            token_speaker : speaker_id for each token
            types: {type : number of tokens}
            speakers: { speaker : number of tokens}
            speaker_types: { speaker: number of clusters the spk appears in}
            type_speakers: {type: number of speakers in the type (cluster)}
        }
        """

        if get_spkid_from_fid is None:
            class MyDict(dict):
                def __missing__(self, key):
                    return key
            get_spkid_from_fid = MyDict()
        tokens = [f for c in clusters for f in c]
        # check that no token is present twice in the list
        nb_unique_tokens = \
            len(np.unique([a+"--"+str(b)+"--"+str(c) for a, b, c in tokens]))
        if len(tokens) != nb_unique_tokens:
            print("Warning : Your dataset has %s duplicates" %
                  (len(tokens) - nb_unique_tokens))
        tokens_type = [i for i, c in enumerate(clusters) for f in c]
        tokens_speaker = [get_spkid_from_fid[f[0]] for f in tokens]
        types = [len(c) for c in clusters]
        speakers = {}
        for spk in np.unique(tokens_speaker):
            speakers[spk] = len(np.where(np.array(tokens_speaker) == spk)[0])
        speakers_types = {spk: 0 for spk in speakers}
        types_speakers = []
        for c in clusters:
            cluster_speakers = np.unique([get_spkid_from_fid[f[0]] for f in c])
            for spk in cluster_speakers:
                speakers_types[spk] = speakers_types[spk]+1
            types_speakers.append(len(cluster_speakers))
        std_descr = {'tokens': tokens,
                     'tokens_type': tokens_type,
                     'tokens_speaker': tokens_speaker,
                     'types': types,
                     'speakers': speakers,
                     'speakers_types': speakers_types,
                     'types_speakers': types_speakers}
        return std_descr

    def type_sample_p(self, std_descr,  type_sampling_mode='log'):
        """
        This function creates the probability matrix for sampling
        a specific cluster.

        For same type, the proba is P(type)
        For different type, the proba is P(type1, type2) = P(type1) * P(type2)


        Sampling proba modes for the types:
            - 1 : equiprobable
            - f2 : proportional to type probabilities
            - f : proportional to square root of type probabilities
            - fcube : proportional to cube root of type probabilities
            - log : proportional to log of type probabilities

        """
        nb_tok = len(std_descr['tokens'])
        tokens_type = std_descr['tokens_type']
        W_types = {}
        nb_types = len(std_descr['types'])

        transfo_error = 'Transformation not implemented'
        list_samplings = ['1', 'f', 'f2', 'log', 'fcube']
        assert type_sampling_mode in list_samplings, transfo_error

        if type_sampling_mode == '1':
            def type_samp_func(x): return 1.0
        if type_sampling_mode == 'f2':
            def type_samp_func(x): return x
        if type_sampling_mode == 'f':
            def type_samp_func(x): return np.sqrt(x)
        if type_sampling_mode == 'fcube':
            def type_samp_func(x): return np.cbrt(x)
        if type_sampling_mode == 'log':
            def type_samp_func(x): return np.log(1+x)

        for tok in range(nb_tok):
            try:
                W_types[tokens_type[tok]] += 1.0
            except Exception as e:
                W_types[tokens_type[tok]] = 1.0

        p_types = dict()

        for type_idx in range(nb_types):
            p_types[type_idx] = type_samp_func(W_types[type_idx])
        return p_types

    def sample_spk_p(self, std_descr, spk_sampling_mode='log'):
        """Sampling proba modes for the speakers conditionned
        by the drawn type(s)
            - 1 : equiprobable
            - f2 : proportional to type probabilities
            - f : proportional to square root of type probabilities
            - fcube : proportional to cube root of type probabilites
            - log : proportional to log of type probabilities
        """
        nb_tok = len(std_descr['tokens'])
        tokens_type = std_descr['tokens_type']
        p_spk_types = {'Stype_Sspk': {}, 'Stype_Dspk': {},
                       'Dtype_Sspk': {}, 'Dtype_Dspk': {}}
        speakers = std_descr['tokens_speaker']
        W_spk_types = {}
        for tok in range(nb_tok):
            try:
                W_spk_types[(speakers[tok], tokens_type[tok])] += 1.0
            except Exception as e:
                W_spk_types[(speakers[tok], tokens_type[tok])] = 1.0

        if spk_sampling_mode == '1':
            def spk_samp_func(x):
                if x == 0:
                    return 0.0
                else:
                    return 1.0
        if spk_sampling_mode == 'f2':
            def spk_samp_func(x): return x
        if spk_sampling_mode == 'f':
            def spk_samp_func(x): return np.sqrt(x)
        if spk_sampling_mode == 'fcube':
            def spk_samp_func(x): return np.cbrt(x)
        if spk_sampling_mode == 'log':
            def spk_samp_func(x): return np.log(1+x)

        print_progress = progress(len(W_spk_types.keys()),
                                  every=0.1,
                                  title="Generate speaker probas")
        i = 0
        for (spk, type_idx) in W_spk_types.keys():
            print_progress(i)
            i += 1
            for (spk2, type_jdx) in W_spk_types.keys():
                if spk == spk2:
                    if type_idx == type_jdx:
                        if (W_spk_types[(spk, type_idx)] - 1) == 0:
                            p_spk_types['Stype_Sspk'][(spk, type_idx)] = 0.0
                        else:
                            p_spk_types['Stype_Sspk'][(spk, type_idx)] = \
                                spk_samp_func(W_spk_types[(spk, type_idx)])
                    else:
                        min_idx = min(type_idx, type_jdx)
                        max_idx = max(type_idx, type_jdx)
                        p_spk_types['Dtype_Sspk'][(spk, min_idx, max_idx)] = \
                            spk_samp_func(W_spk_types[(spk, type_idx)]) * \
                            spk_samp_func(W_spk_types[(spk, type_jdx)])
                else:
                    if type_idx == type_jdx:
                        p_spk_types['Stype_Dspk'][(spk, spk2, type_idx)] = \
                            spk_samp_func(W_spk_types[(spk, type_idx)]) * \
                            spk_samp_func(W_spk_types[(spk2, type_idx)])
                    else:
                        min_idx = min(type_idx, type_jdx)
                        max_idx = max(type_idx, type_jdx)
                        p_spk_types['Dtype_Dspk'][(spk, spk2,
                                                   min_idx, max_idx)] = \
                            spk_samp_func(W_spk_types[(spk, type_idx)]) * \
                            spk_samp_func(W_spk_types[(spk2, type_jdx)])
        return p_spk_types

    def generate_token_dict(self, std_descr):
        tokens = defaultdict(list)
        nb_tok = len(std_descr['tokens'])
        speakers = std_descr['tokens_speaker']
        types = std_descr['tokens_type']

        for tok_id in range(nb_tok):
            tokens[(types[tok_id], speakers[tok_id])].append(tok_id)

        return tokens

    def type_speaker_sampling_p(self, std_descr=None,
                                type_sampling_mode='f', spk_sampling_mode='f'):
        """
        This function generates the final probability matrix
        P(type, speaker)
        We have 4 different matrices :
        St, Ss : P(type, speaker)
        St, Ds : P(type, sp1, sp2)
        Dt, Ss : P(t1, t2, s)
        Dt, Ds : P(t1, t2, s1, s2)

        This is computed using bayes rules :
        P(type, speaker) = P(type) * P(speaker | type)

        Sampling proba modes for p_i1,i2,j1,j2
            It is based on Bayes rule:
                - log : proporitonal to log of speaker or type probabilities
                - f : proportional to square roots of speaker
                    or type probabilities (in order to obtain sampling
                    probas for pairs proportional to geometric mean of
                    the members of the pair probabilities)
                - f2 : proportional to speaker
                    or type probabilities
                - fcube: proportionnal to cubic root of probabilities
                - 1 : equiprobable

            Parameters
            ----------
            std_descr : dict
                dictionnary with all description of the clusters
            type_sampling_mode : String
                function applied to the observed type frequencies
            spk_sampling_mode : String
                function applied to the observed speaker frequencies

        """
        assert type_sampling_mode in ['1', 'f', 'f2', 'log', 'fcube']
        assert spk_sampling_mode in ['1', 'f', 'f2', 'log', 'fcube']
        # W_types = std_descr['types']
        # speakers = [e for e in std_descr['speakers']]
        # W_speakers = [std_descr['speakers'][e] for e in speakers]
        p_types = self.type_sample_p(std_descr,
                                     type_sampling_mode=type_sampling_mode)
        p_spk_types = self.sample_spk_p(std_descr,
                                        spk_sampling_mode=spk_sampling_mode)

        p_types = normalize_distribution(p_types)

        for config in p_spk_types.keys():
            p_spk_types[config] = normalize_distribution(p_spk_types[config])

        print_progress = progress(len(p_spk_types.keys()),
                                  every=0.01,
                                  title="Generate type-speaker probas")
        i = 0
        for config in p_spk_types.keys():
            print_progress(i)
            i += 1
            if config == 'Stype_Sspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = \
                        p_types[el[1]] * \
                        p_spk_types[config][el]
            if config == 'Stype_Dspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = \
                        p_types[el[2]] * \
                        p_spk_types[config][el]
            if config == 'Dtype_Sspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = \
                        p_types[el[1]] * \
                        p_types[el[2]] * \
                        p_spk_types[config][el]
            if config == 'Dtype_Dspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = \
                        p_types[el[2]] * \
                        p_types[el[3]] * \
                        p_spk_types[config][el]

        for config in p_spk_types.keys():
            p_spk_types[config] = normalize_distribution(p_spk_types[config])

        return p_spk_types

    def compute_cdf(self, proba):
        cdf = {}
        for key in proba.keys():
            cdf[key] = cumulative_distribution(proba[key])
        return cdf


class SamplerClusterSiamese(SamplerCluster):
    """Sampler for Siamese network based on clusters of words

    """

    def __init__(self, *args, **kwargs):
        super(SamplerClusterSiamese, self).__init__(*args, **kwargs)

    def whoami(self):
        return {'params': self.__dict__, 'class_name': self.__class__.__name__}

    def sample_batch(self,
                     p_spk_types,
                     cdf,
                     token_dict,
                     num_samples=5012):

        """Sampling proba modes for p_i1,i2,j1,j2
            It is based on Bayes rule:
                - log : proporitonal to log of speaker or type probabilities
                - f : proportional to square roots of speaker
                    or type probabilities (in order to obtain sampling
                    probas for pairs proportional to geometric mean of
                    the members of the pair probabilities)
                - f2 : proportional to speaker
                    or type probabilities
                - fcube: proportionnal to log probabilities
                - 1 : equiprobable

            Parameters
            ----------
            p_spk_types : dict
                dictionnary with the probabilites of all different config
            cdf : dict
                dictionnary with the Cumulative distribution computed for
                the different configurations
            pairs : dict
                dictionnary with the possible pairs
            seed : int
                seed
            num_samples : int
                number of pairs to compute
            ratio_same_diff_spk : float
                float between 0 and 1, percentage of different speaker pairs
                fed to the ABnet3
            ratio_same_diff_type : float
                float between 0 and 1, percentage of different phn pairs
                fed to the ABnet3
        """
        np.random.seed(self.seed)
        sampled_tokens = {
                          'Stype_Sspk': [],
                          'Stype_Dspk': [],
                          'Dtype_Sspk': [],
                          'Dtype_Dspk': []
                          }
        num_same_spk = int((num_samples) * (1 - self.ratio_same_diff_spk))
        num_diff_spk = num_samples - num_same_spk
        num_Stype_Sspk = int(num_same_spk*(1-self.ratio_same_diff_type))
        num_Dtype_Sspk = int(num_same_spk*(self.ratio_same_diff_type))
        num_Stype_Dspk = int(num_diff_spk*(1-self.ratio_same_diff_type))
        num_Dtype_Dspk = int(num_diff_spk*(self.ratio_same_diff_type))
        sampled_ratio = {
                         'Stype_Sspk': num_Stype_Sspk,
                         'Stype_Dspk': num_Stype_Dspk,
                         'Dtype_Sspk': num_Dtype_Sspk,
                         'Dtype_Dspk': num_Dtype_Dspk
                         }
        for config in p_spk_types.keys():
            keys = np.array(list(p_spk_types[config].keys()))
            sample_idx = sample_searchidx(cdf[config], sampled_ratio[config])
            sample = keys[sample_idx]
            if config == 'Stype_Sspk':
                for key in sample:
                    spk, type_idx = key
                    tokens = token_dict[int(type_idx), spk]
                    tok1, tok2 = np.random.choice(tokens, size=2,
                                                  replace=False)
                    sampled_tokens[config].append(
                        (tok1, tok2))
            if config == 'Stype_Dspk':
                for key in sample:
                    spk1, spk2, type_idx = key
                    type_idx = int(type_idx)
                    tok1 = np.random.choice(token_dict[type_idx, spk1])
                    tok2 = np.random.choice(token_dict[type_idx, spk2])
                    sampled_tokens[config].append((tok1, tok2))
            if config == 'Dtype_Sspk':
                for key in sample:
                    spk, type_idx, type_jdx = key
                    type_idx = int(type_idx)
                    type_jdx = int(type_jdx)
                    tok1 = np.random.choice(token_dict[type_idx, spk])
                    tok2 = np.random.choice(token_dict[type_jdx, spk])
                    sampled_tokens[config].append((tok1, tok2))
            if config == 'Dtype_Dspk':
                for key in sample:
                    spk1, spk2, type_idx, type_jdx = key
                    type_idx = int(type_idx)
                    type_jdx = int(type_jdx)
                    try:
                        tok1 = np.random.choice(token_dict[type_idx, spk1])
                        tok2 = np.random.choice(token_dict[type_jdx, spk2])
                    except Exception:
                        tok1 = np.random.choice(token_dict[type_idx, spk2])
                        tok2 = np.random.choice(token_dict[type_jdx, spk1])
                    sampled_tokens[config].append((tok1, tok2))
        return sampled_tokens

    def write_tokens(self, descr=None, proba=None, cdf=None,
                     token_dict=None, batch_size=8, num_samples=0,
                     out_dir=None, seed=0):
        """Write tokens based on all different parameters and write the tokens
        in a batch.

        """
        lines = []
        np.random.seed(seed)
        print("Sampling tokens")

        sampled_batch = self.sample_batch(proba, cdf, token_dict,
                                          num_samples=num_samples)
        for config in sampled_batch.keys():
            if config == 'Stype_Sspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")
            if config == 'Stype_Dspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")
            if config == 'Dtype_Sspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")
            if config == 'Dtype_Dspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")

        np.random.shuffle(lines)
        # prev_idx = 0
        print("Writing tokens to disk")
        if self.sample_batches:
            for idx in range(1, int(num_samples // batch_size)):
                with open(os.path.join(out_dir, 'pair_' +
                          str(idx)+'.batch'), 'w') as fh:
                        curr_lines = lines[(idx-1)*batch_size:(idx)*batch_size]
                        fh.writelines(curr_lines)
        else:
            text = "".join(lines)
            with open(os.path.join(out_dir, 'dataset'), 'w') as fh:
                fh.write(text)
            print("done write_tokens")

    def export_pairs(self, out_dir=None,
                     descr=None, type_sampling_mode='',
                     spk_sampling_mode='',
                     seed=0, batch_size=8, num_samples=None):
        np.random.seed(seed)
        same_pairs = ['Stype_Sspk', 'Stype_Dspk']
        diff_pairs = ['Dtype_Sspk', 'Dtype_Dspk']
        token_dict = self.generate_token_dict(descr)
        proba = self.type_speaker_sampling_p(
                    std_descr=descr,
                    type_sampling_mode=type_sampling_mode,
                    spk_sampling_mode=spk_sampling_mode)

        print("Cumulative distribution")
        cdf = {}
        for key in proba.keys():
            cdf[key] = cumulative_distribution(proba[key])

        # This computation is important for the total number of batches/pairs
        # Number of possible pairs in the smallest count
        # of different words for a speaker
        num = np.min(list(descr['speakers'].values()))
        if num_samples is None:
            num_samples = num*(num-1)/2
        idx_batch = 0
        self.write_tokens(descr=descr, proba=proba, cdf=cdf,
                          token_dict=token_dict, batch_size=self.batch_size,
                          num_samples=num_samples, out_dir=out_dir, seed=seed)
        print("done export_pairs")

    def sample(self):
        """
        Main function : takes Term Discovery results and sample pairs
            for training and dev set to train ABnet from it.
        Parameters
        ----------
        std_file : String
            Term Discovery results (.class file)
        spkid_file : String
            Mapping between wav_id and spk_id
        out_dir : String
            target directory for output files
        stats : Bool
            TODO: not implemented yet
            if True the function outputs 'figures.pdf' containing
                some stats on the Term Discovery results;
                otherwise the function outputs files train.pairs and
                dev.pairs containing the sampled pairs
        seed : Int
            random seed
        type_sampling_mode : String
            sampling mode for token types ('1', 'f','f2','fcube' or 'log')
        spk_sampling_mode : String
            sampling mode for token speakers ('1', 'f','f2','fcube' or 'log')
        batch_size : Int
            number of pairs per batch,
        """

        # 0) Read mapping for id to speaker
        print("Reading id to speaker file %s" % self.spkid_file)
        get_spkid_from_fid = read_spkid_file(self.spkid_file)

        # 1) parsing files to get clusters and speakers
        print("Reading cluster file %s with max_num_clusters = %s" %
              (self.std_file, self.max_num_clusters))
        clusters = self.parse_input_file(self.std_file, self.max_num_clusters)
        print("We have %s clusters." % len(clusters))
        if self.spk_list_file is not None:
            print("Reading speaker list file %s" % self.spk_list_file)
            spk_list = read_spk_list(self.spk_list_file)

        # 2) Split the clusters according to train/dev ratio
        split_clusters = self.split_clusters_ratio(clusters)
        train_clusters, dev_clusters = split_clusters

        # 3) Analysis of clusters to be able to sample
        train_descr = self.analyze_clusters(train_clusters, get_spkid_from_fid)
        dev_descr = self.analyze_clusters(dev_clusters, get_spkid_from_fid)

        # train and dev stats
        # if stats:
        #     try:
        #         #TODO remake a plot_stats function with new functions.
        #         pdf_file = os.path.join(out_dir, 'figures.pdf')
        #         pdf_pages = PdfPages(pdf_file)
        #         plot_stats(std_descr, 'Whole data', pdf_pages)
        #         plot_stats(train_descr, 'Train set', pdf_pages)
        #         plot_stats(test_descr, 'Test set', pdf_pages)
        #     finally:
        #         pdf_pages.close()
        # else:
        # generate and write pairs to disk

        os.makedirs(self.directory_output)
        # 4) Make directory and export pairs to the disk
        train_pairs_dir = os.path.join(self.directory_output, 'train_pairs')
        os.makedirs(os.path.join(self.directory_output, 'train_pairs'))

        if self.num_total_sampled_pairs is not None:
            prop_train = self.num_total_sampled_pairs * self.ratio_train_dev
            num_samples_train = int(prop_train)
            num_samples_dev = self.num_total_sampled_pairs - num_samples_train
        else:
            num_samples_train, num_samples_dev = None, None

        self.export_pairs(out_dir=train_pairs_dir,
                          descr=train_descr,
                          type_sampling_mode=self.type_sampling_mode,
                          spk_sampling_mode=self.spk_sampling_mode,
                          seed=self.seed, batch_size=self.batch_size,
                          num_samples=num_samples_train)
        dev_pairs_dir = os.path.join(self.directory_output, 'dev_pairs')
        print("Done writing training pairs")
        os.makedirs(dev_pairs_dir)
        self.export_pairs(out_dir=dev_pairs_dir,
                          descr=dev_descr,
                          type_sampling_mode=self.type_sampling_mode,
                          spk_sampling_mode=self.spk_sampling_mode,
                          seed=self.seed+1, batch_size=self.batch_size,
                          num_samples=num_samples_dev)
        print("Done writing dev pairs")


if __name__ == '__main__':

    input_file = '/Users/rachine/abnet3/english.wrd.classes'
    batch_size = 8
    directory_output = '/Users/rachine/abnet3/results/test_pairs_sampler'
    seed = 0
    run = 'once'
    max_clusters = 2000
    type_sampling_mode = 'log'
    spk_sampling_mode = 'log'

    sam = SamplerClusterSiamese(std_file=input_file, batch_size=batch_size,
                                run=run, seed=seed,
                                type_sampling_mode=type_sampling_mode,
                                spk_sampling_mode=spk_sampling_mode,
                                max_clusters=max_clusters,
                                directory_output=directory_output)
