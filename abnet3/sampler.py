#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script is composed of the different modules for sampling pairs
based on gold words or spoken term discovery outputs.

It will generate text files in two folders 'train_pairs' and 'dev_pairs' in the
correct format for the Neural Network.

"""

from abnet3.utils import normalize_distribution, cumulative_distribution
from abnet3.utils import print_token, sample_searchidx
from abnet3.utils import read_spkid_file, read_spk_list

import numpy as np
import os
import codecs


class SamplerBuilder(object):
    """Sampler Model interface

        Parameters
        ----------
        input_file : String
            Path to clusters of words
        batch_size : Int
            Number of words per batch
        already_done : Bool
            If already sampled, specify it
        directory_output : String
            Path folder where train/dev pairs folder will be
        seed : int
            Seed

    """
    def __init__(self, batch_size=8, already_done=False, input_file=None,
                 directory_output=None, ratio_train_dev=0.7, seed=0):
        super(SamplerBuilder, self).__init__()
        self.input_file = input_file
        self.batch_size = batch_size
        self.already_done = already_done
        self.directory_output = directory_output
        self.seed = seed
        self.ratio_train_dev = ratio_train_dev

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

    """
    def __init__(self, max_size_cluster=10, ratio_same_diff_spk=0.25,
                 type_samp='log', spk_samp='log',
                 std_file=None, spk_list_file=None, spkid_file=None,
                 *args, **kwargs):
        super(SamplerCluster, self).__init__(*args, **kwargs)
        self.max_size_cluster = max_size_cluster
        self.ratio_same_diff_spk = ratio_same_diff_spk
        self.type_sampling_mode = type_sampling_mode
        self.spk_sampling_mode = spk_sampling_mode
        self.std_file = std_file
        self.spk_list_file = spk_list_file
        self.spkid_file = spkid_file

    def parse_input_file(self, input_file=None, max_clusters=-1):
        """Parse input file:

        Parameters
        ----------
        input_file : String
            Path to clusters of words
        max_size_cluster : Int
            Number max of clusters, useful for debugging
        """
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

        return clusters

    def split_clusters_ratio(self, clusters,
                             get_spkid_from_fid=None):
        """Split clusters, two type of splits involved for the train and
            dev. Biggest clusters are going to be split with the ratio and
            the result parts goes to train and dev. The other smaller
            clusters are then chosen randomly to go either in the train or dev.

            Parameters
            ----------
            clusters : list
                List of cluster parsed from parse_input_file
            get_spkid_from_fid : dict
                Mapping dictionnary between files and speaker id
            ratio_train_dev: float
                Ratio number between 0 and 1 to randomly split train and
                dev clusters
        """

        train_clusters, dev_clusters = [], []
        size_clusters = np.array([len(cluster) for cluster in clusters])
        new_clusters = []
        num_clusters = len(clusters)
        num_train = int(self.ratio_train_dev*num_clusters)
        train_idx = np.random.choice(num_clusters, num_train, replace=False)

        for idx, cluster in enumerate(clusters):
            train_cluster = [tok for tok in cluster
                             if idx in train_idx]
            dev_cluster = [tok for tok in cluster
                           if idx not in train_idx]
            if train_cluster:
                # Tricky move here to split big clusters for train and dev
                size_cluster = len(train_cluster)
                if self.max_size_cluster > 1 and \
                   self.max_size_cluster < size_cluster:
                    num_train = int(self.ratio_train_dev*size_cluster)
                    indexes = range(size_cluster)
                    rand_idx = np.random.permutation(indexes)
                    train_clusters.append(train_cluster[rand_idx[:num_train]])
                    dev_clusters.append(train_cluster[rand_idx[num_train:]])
                else:
                    train_clusters.append(train_cluster)
            if dev_cluster:
                dev_clusters.append(dev_cluster)
        return train_clusters, dev_clusters

    def analyze_clusters(self, clusters, get_spkid_from_fid=None):
        """Analysis input file to prepare sampler

        Parameters
        ----------
        clusters : list
            List of cluster parsed from parse_input_file
        get_spkid_from_fid : dict
            Mapping dictionnary between files and speaker id

        """

        if get_spkid_from_fid is None:
            def get_spkid_from_fid(x): return x
        tokens = [f for c in clusters for f in c]
        # check that no token is present twice in the list
        nb_unique_tokens = \
            len(np.unique([a+"--"+str(b)+"--"+str(c) for a, b, c in tokens]))
        assert len(tokens) == nb_unique_tokens
        tokens_type = [i for i, c in enumerate(clusters) for f in c]
        tokens_speaker = [get_spkid_from_fid(f[0]) for f in tokens]
        types = [len(c) for c in clusters]
        speakers = {}
        for spk in np.unique(tokens_speaker):
            speakers[spk] = len(np.where(np.array(tokens_speaker) == spk)[0])
        speakers_types = {spk: 0 for spk in speakers}
        types_speakers = []
        for c in clusters:
            cluster_speakers = np.unique([get_spkid_from_fid(f[0]) for f in c])
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

    def type_sample_p(self, std_descr,  type_samp='log'):
        """Sampling proba modes for the types:
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
        assert type_samp in ['1', 'f', 'f2', 'log', 'fcube'], transfo_error

        if type_samp == '1':
            def type_samp_func(x): 1.0
        if type_samp == 'f2':
            def type_samp_func(x): x
        if type_samp == 'f':
            def type_samp_func(x): np.sqrt(x)
        if type_samp == 'fcube':
            def type_samp_func(x): np.cbrt(x)
        if type_samp == 'log':
            def type_samp_func(x): np.log(1+x)

        for tok in range(nb_tok):
            try:
                W_types[tokens_type[tok]] += 1.0
            except e:
                W_types[tokens_type[tok]] = 1.0
        p_types = {"Stype": {}, "Dtype": {}}

        for type_idx in range(nb_types):
            p_types["Stype"][type_idx] = type_samp_func(W_types[type_idx])
            for type_jdx in range(type_idx+1, nb_types):
                p_types["Dtype"][(type_idx, type_jdx)] = \
                    type_samp_func(W_types[type_idx]) * \
                    type_samp_func(W_types[type_jdx])
        return p_types

    def sample_spk_p(self, std_descr, spk_samp='log'):
        """Sampling proba modes for the speakers conditionned by the drawn type(s)
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
            except e:
                W_spk_types[(speakers[tok], tokens_type[tok])] = 1.0

        if spk_samp == '1':
            def spk_samp_func(x): 0.0 if x == 0 else 1.0
        if spk_samp == 'f2':
            def spk_samp_func(x): x
        if spk_samp == 'f':
            def spk_samp_func(x): np.sqrt(x)
        if spk_samp == 'fcube':
            def spk_samp_func(x): np.cbrt(x)
        if spk_samp == 'log':
            def spk_samp_func(x): np.log(1+x)

        # nb_types = len(std_descr['types'])
        for (spk, type_idx) in W_spk_types.keys():
            for (spk2, type_jdx) in W_spk_types.keys():
                if spk == spk2:
                    if type_idx == type_jdx:
                        if (W_spk_types[(spk, type_idx)] - 1) == 0:
                            p_spk_types['Stype_Sspk'][(spk, type_idx)] = 0.0
                        else:
                            p_spk_types['Stype_Sspk'][(spk, type_idx)] = \
                                spk_samp_func(W_spk_types[(spk, type_idx)])
                    else:
                        min_idx = np.min([type_idx, type_jdx])
                        max_idx = np.max([type_idx, type_jdx])
                        p_spk_types['Dtype_Sspk'][(spk, min_idx, max_idx)] = \
                            spk_samp_func(W_spk_types[(spk, type_idx)]) * \
                            spk_samp_func(W_spk_types[(spk, type_jdx)])
                else:
                    if type_idx == type_jdx:
                        p_spk_types['Stype_Dspk'][(spk, spk2, type_idx)] = \
                            spk_samp_func(W_spk_types[(spk, type_idx)]) * \
                            spk_samp_func(W_spk_types[(spk2, type_idx)])
                    else:
                        min_idx = np.min([type_idx, type_jdx])
                        max_idx = np.max([type_idx, type_jdx])
                        p_spk_types['Dtype_Dspk'][(spk, spk2,
                                                   min_idx, max_idx)] = \
                            spk_samp_func(W_spk_types[(spk, type_idx)]) * \
                            spk_samp_func(W_spk_types[(spk2, type_jdx)])
        return p_spk_types

    def generate_possibilities(self, std_descr):
        """Generate possibilities between (types,speakers) and tokens/realisations

        """
        pairs = {'Stype_Sspk': {},
                 'Stype_Dspk': {},
                 'Dtype_Sspk': {},
                 'Dtype_Dspk': {}}
        nb_tok = len(std_descr['tokens'])
        speakers = std_descr['tokens_speaker']
        types = std_descr['tokens_type']
        for tok1 in range(nb_tok):
            for tok2 in range(tok1+1, nb_tok):
                spk_type = 'S' if speakers[tok1] == speakers[tok2] else 'D'
                type_type = 'S' if types[tok1] == types[tok2] else 'D'
                pair_type = type_type + 'type_' + spk_type + 'spk'
                try:
                    if pair_type == 'Stype_Sspk':
                        pairs[pair_type][
                                        (speakers[tok1],
                                         types[tok1])].append((tok1, tok2))
                    if pair_type == 'Stype_Dspk':
                        pairs[pair_type][
                                         (speakers[tok1], speakers[tok2],
                                          types[tok1])].append((tok1, tok2))
                    if pair_type == 'Dtype_Sspk':
                        min_idx = np.min([types[tok1], types[tok2]])
                        max_idx = np.max([types[tok1], types[tok2]])
                        pairs[pair_type][
                                         (speakers[tok1], min_idx,
                                          max_idx)].append((tok1, tok2))
                    if pair_type == 'Dtype_Dspk':
                        min_idx = np.min(types[tok1], types[tok2])
                        max_idx = np.max([types[tok1], types[tok2]])
                        pairs[pair_type][
                                         (speakers[tok1], speakers[tok2],
                                          min_idx, max_idx)
                                         ].append((tok1, tok2))
                except e:
                    if pair_type == 'Stype_Sspk':
                        pairs[pair_type][
                                         (speakers[tok1],
                                          types[tok1])] = [(tok1, tok2)]
                    if pair_type == 'Stype_Dspk':
                        pairs[pair_type][(speakers[tok1], speakers[tok2],
                                          types[tok1])] = [(tok1, tok2)]
                    if pair_type == 'Dtype_Sspk':
                        min_idx = np.min([types[tok1], types[tok2]])
                        max_idx = np.max([types[tok1], types[tok2]])
                        pairs[pair_type][
                                         (speakers[tok1], min_idx,
                                          max_idx)
                                          ] = [(tok1, tok2)]
                    if pair_type == 'Dtype_Dspk':
                        min_idx = np.min([types[tok1], types[tok2]])
                        max_idx = np.max([types[tok1], types[tok2]])
                        pairs[pair_type][
                                         (speakers[tok1], speakers[tok2],
                                          min_idx, max_idx)
                                         ] = [(tok1, tok2)]
        return pairs

    def type_speaker_sampling_p(self, std_descr=None,
                                type_samp='f', speaker_samp='f'):
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
            std_descr : dict
                dictionnary with all description of the clusters
            type_samp : String
                function applied to the observed type frequencies
            spk_samp : String
                function applied to the observed speaker frequencies

        """
        assert type_samp in ['1', 'f', 'f2', 'log', 'fcube']
        assert speaker_samp in ['1', 'f', 'f2', 'log', 'fcube']
        # W_types = std_descr['types']
        # speakers = [e for e in std_descr['speakers']]
        # W_speakers = [std_descr['speakers'][e] for e in speakers]
        p_types = self.type_sample_p(std_descr,  type_samp=type_samp)
        p_spk_types = self.sample_spk_p(std_descr, spk_samp=speaker_samp)

        for config in p_types.keys():
            p_types[config] = normalize_distribution(p_types[config])

        for config in p_spk_types.keys():
            p_spk_types[config] = normalize_distribution(p_spk_types[config])

        for config in p_spk_types.keys():
            if config == 'Stype_Sspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Stype'][el[1]] * \
                        p_spk_types[config][el]
            if config == 'Stype_Dspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Stype'][el[2]] * \
                        p_spk_types[config][el]
            if config == 'Dtype_Sspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Dtype'][
                                                               (el[1],
                                                                el[2])] * \
                        p_spk_types[config][el]
            if config == 'Dtype_Dspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Dtype'][
                                                               (el[2],
                                                                el[3])] * \
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
                     pairs,
                     num_examples=5012):

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
            num_examples : int
                number of pairs to compute
            ratio_same_diff_spk : float
                float between 0 and 1 which is the ration of same and different
                speaker for the pairs fed to the ABnet3
        """
        np.random.seed(self.seed)
        sampled_tokens = {
                          'Stype_Sspk': [],
                          'Stype_Dspk': [],
                          'Dtype_Sspk': [],
                          'Dtype_Dspk': []
                          }
        num_same_spk = int((num_examples)*self.ratio_same_diff_spk)
        num_diff_spk = num_examples - num_same_spk
        sampled_ratio = {
                         'Stype_Sspk': num_same_spk/2,
                         'Stype_Dspk': num_diff_spk/2,
                         'Dtype_Sspk': num_same_spk/2,
                         'Dtype_Dspk': num_diff_spk/2
                         }
        for config in p_spk_types.keys():
            # proba_config = np.array(p_spk_types[config].values())
            # sizes = len(p_spk_types[config].keys())
            keys = np.array(p_spk_types[config].keys())
            sample_idx = sample_searchidx(cdf[config], sampled_ratio[config])
            sample = keys[sample_idx]
            if config == 'Stype_Sspk':
                for key in sample:
                    spk, type_idx = key
                    pot_tok = pairs[config][spk, int(type_idx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(
                        pot_tok[np.random.choice(num_tok)])
            if config == 'Stype_Dspk':
                for key in sample:
                    spk1, spk2, type_idx = key
                    try:
                        pot_tok = pairs[config][spk1, spk2, int(type_idx)]
                    except e:
                        pot_tok = pairs[config][spk2, spk1, int(type_idx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(
                        pot_tok[np.random.choice(num_tok)])
            if config == 'Dtype_Sspk':
                for key in sample:
                    spk, type_idx, type_jdx = key
                    pot_tok = pairs[config][spk, int(type_idx), int(type_jdx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(
                        pot_tok[np.random.choice(num_tok)])
            if config == 'Dtype_Dspk':
                for key in sample:
                    spk1, spk2, type_idx, type_jdx = key
                    try:
                        pot_tok = pairs[config][spk1, spk2,
                                                int(type_idx), int(type_jdx)]
                    except e:
                        pot_tok = pairs[config][spk2, spk1,
                                                int(type_idx), int(type_jdx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(
                        pot_tok[np.random.choice(num_tok)])
        return sampled_tokens

    def write_tokens(self, descr=None, proba=None, cdf=None,
                     pairs={}, size_batch=8, num_batches=0,
                     out_dir=None, seed=0):
        """Write tokens based on all different parameters and write the tokens
        in a batch.

        """
        lines = []
        np.random.seed(seed)
        sampled_batch = self.sample_batch(proba, cdf, pairs,
                                          num_pairs_batch=num_batches)
        for config in sampled_batch.keys():
            if config == 'Stype_Sspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")
            if config == 'Stype_Dspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")
            if config == 'Dtype_Sspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")
            if config == 'Dtype_Dspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " + pair_type + "\n")

        np.random.shuffle(lines)
        # prev_idx = 0
        for idx in range(1, num_batches//size_batch):
            with open(os.path.join(out_dir, 'pair_' +
                      str(idx)+'.batch', 'w')) as fh:
                    fh.writelines(lines[(idx-1)*size_batch:(idx)*size_batch])

    def export_pairs(self, out_dir=None,
                     descr=None, type_sampling_mode='',
                     spk_sampling_mode='',
                     seed=0, size_batch=8):
        np.random.seed(seed)
        same_pairs = ['Stype_Sspk', 'Stype_Dspk']
        diff_pairs = ['Dtype_Sspk', 'Dtype_Dspk']
        pairs = self.generate_possibilities(descr)
        proba = self.type_speaker_sampling_p(std_descr=descr,
                                             type_samp=type_sampling_mode,
                                             speaker_samp=spk_sampling_mode)
        cdf = {}
        for key in proba.keys():
            cdf[key] = cumulative_distribution(proba[key])

        num = np.min(descr['speakers'].values())
        num_batches = num*(num-1)/2
        idx_batch = 0
        self.write_tokens(descr=descr, proba=proba, cdf=cdf,
                          pairs=pairs, size_batch=self.size_batch,
                          num_batches=num_batches, out_dir=out_dir, seed=seed)

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
        size_batch : Int
            number of pairs per batch,
        """

        # 0) Read mapping for id to speaker
        get_spkid_from_fid = read_spkid_file(self.spkid_file)

        # 1) parsing files to get clusters and speakers
        clusters = self.parse_STD_results(self.std_file)
        spk_list = read_spk_list(self.spk_list_file)

        # 2) Split the clusters according to train/dev ratio
        split_clusters = self.split_clusters_ratio(clusters, spk_list,
                                                   get_spkid_from_fid)
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

        # 4) Make directory and export pairs to the disk
        train_pairs_dir = os.path.join(self.directory_output, 'train_pairs')
        os.makedirs(os.path.join(self.directory_output, 'train_pairs'))
        self.export_pairs(out_dir=train_pairs_dir,
                          descr=train_descr,
                          type_sampling_mode=self.type_sampling_mode,
                          spk_sampling_mode=spk_sampling_mode,
                          seed=seed, size_batch=size_batch)
        dev_pairs_dir = os.path.join(self.directory_output, 'dev_pairs')
        os.makedirs(dev_pairs_dir)
        self.export_pairs(out_dir=dev_pairs_dir,
                          descr=dev_descr,
                          type_sampling_mode=self.type_sampling_mode,
                          spk_sampling_mode=self.spk_sampling_mode,
                          seed=self.seed+1, size_batch=self.size_batch)


if __name__ == '__main__':

    input_file = '/Users/rachine/abnet3/english.wrd.classes'
    batch_size = 8
    directory_output = '/Users/rachine/abnet3/results/test_pairs_sampler'
    seed = 0
    already_done = False
    max_clusters = 2000
    type_samp = 'log'
    spk_samp = 'log'

    sam = SamplerClusterSiamese(input_file=input_file, batch_size=batch_size,
                                already_done=already_done, seed=seed,
                                type_samp=type_samp, spk_samp=spk_samp,
                                max_clusters=max_clusters,
                                directory_output=directory_output)
