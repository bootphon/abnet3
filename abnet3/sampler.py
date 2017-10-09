#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:22:41 2017

@author: Rachid Riad
"""

import abnet3.utils
import numpy as np
import os

class SamplerBuilder(object):
    """Sampler Model interface
    
    """
    def __init__(self, batch_size=8, already_done=False, directory_output=None):
        super(SamplerBuilder, self).__init__()
        self.batch_size = batch_size
        self.already_done = already_done
        self.directory_output = directory_output
        
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
    def __init__(self):
        super(SamplerPairs, self).__init__()        


class SamplerCluster(SamplerBuilder):
    """Sampler Model interface based on clusters of words
    
    """
    def __init__(self):
        super(SamplerCluster, self).__init__()        
        
    def parse_input_file(self, input_file=None):
        with open(input_file, 'r') as fh:
            lines = fh.readlines()
        clusters = []
        i = 0
        while i < len(lines):
            cluster = []
            tokens = lines[i].strip().split(" ")
            assert len(tokens) == 2, str(tokens)
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
    
    def analysis_description(self, clusters, get_spkid_from_fid=None):
        """Analysis input file to prepare sampler
        
        """
        if get_spkid_from_fid is None:
            get_spkid_from_fid = lambda x: x
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
        speakers_types = {spk : 0 for spk in speakers}
        types_speakers = []
        for c in clusters:
            cluster_speakers = np.unique([get_spkid_from_fid(f[0]) for f in c])
            for spk in cluster_speakers:
                speakers_types[spk] = speakers_types[spk]+1
            types_speakers.append(len(cluster_speakers))   
        std_descr = {'tokens' : tokens,
                     'tokens_type' : tokens_type,
                     'tokens_speaker' : tokens_speaker,
                     'types' : types,
                     'speakers' : speakers,
                     'speakers_types' : speakers_types,
                     'types_speakers' : types_speakers}
        return std_descr
    
    
    def type_sample_p(self, std_descr,  type_samp = 'log'):
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
        
        assert type_samp in ['1', 'f', 'f2','log', 'fcube'], 'Transformation not implemented'
        
        if type_samp == '1':        
            type_samp_func = lambda x : 1
        if type_samp == 'f2':
            type_samp_func = lambda x : x
        if type_samp == 'f':
            type_samp_func = lambda x : np.sqrt(x)
        if type_samp == 'fcube':
            type_samp_func = lambda x : np.cbrt(x)
        if type_samp == 'log':
            type_samp_func = lambda x : np.log(1+x)
    
        for tok in range(nb_tok):
            try:
                W_types[tokens_type[tok]] += 1.0
            except:
                W_types[tokens_type[tok]] = 1.0
        p_types = {"Stype" : {}, "Dtype":{}}
        
        for type_idx in range(nb_types):
            p_types["Stype"][type_idx] = type_samp_func(W_types[type_idx])
            for type_jdx in range(type_idx+1,nb_types):
                p_types["Dtype"][(type_idx,type_jdx)] = type_samp_func(W_types[type_idx])*type_samp_func(W_types[type_jdx])
        return p_types 

    def sample_spk_p(self, std_descr, spk_samp = 'log'):
        """Sampling proba modes for the speakers conditionned by the drawn type(s)
            - 1 : equiprobable
            - f2 : proportional to type probabilities
            - f : proportional to square root of type probabilities
            - fcube : proportional to cube root of type probabilites
            - log : proportional to log of type probabilities
        """
        nb_tok = len(std_descr['tokens'])
        tokens_type = std_descr['tokens_type']
        p_spk_types = {'Stype_Sspk' : {}, 'Stype_Dspk' : {}, 'Dtype_Sspk' : {}, 'Dtype_Dspk' : {}}
        speakers = std_descr['tokens_speaker'] 
        #list_speakers = std_descr['speakers_types'].keys()
        W_spk_types = {} 
        for tok in range(nb_tok):
            try:
                W_spk_types[(speakers[tok],tokens_type[tok])] += 1.0
            except:
                W_spk_types[(speakers[tok],tokens_type[tok])] = 1.0
        
        if spk_samp == '1':
            spk_samp_func = lambda x : 1.0
        if spk_samp == 'f2':
            spk_samp_func = lambda x : x
        if spk_samp == 'f':
            spk_samp_func = lambda x : np.sqrt(x)
        if spk_samp == 'fcube':
            spk_samp_func = lambda x : np.cbrt(x)
        if spk_samp == 'log':
            spk_samp_func = lambda x : np.log(1+x)
            
        #nb_types = len(std_descr['types'])
        for (spk,type_idx) in W_spk_types.keys(): 
            for (spk2,type_jdx) in W_spk_types.keys():
                if spk == spk2:
                    if type_idx == type_jdx: 
                        if (W_spk_types[(spk,type_idx)] - 1) == 0:
                            p_spk_types['Stype_Sspk'][(spk,type_idx)] = 0.0
                        else:
                            p_spk_types['Stype_Sspk'][(spk,type_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)] )  
                    else:
                        min_idx,max_idx = np.min([type_idx,type_jdx]),np.max([type_idx,type_jdx])
                        p_spk_types['Dtype_Sspk'][(spk,min_idx,max_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)])*spk_samp_func(W_spk_types[(spk,type_jdx)]) 
                else:
                    if type_idx == type_jdx:
                        p_spk_types['Stype_Dspk'][(spk,spk2,type_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)])*spk_samp_func(W_spk_types[(spk2,type_idx)]) 
                    else:
                        min_idx,max_idx = np.min([type_idx,type_jdx]),np.max([type_idx,type_jdx])
                        p_spk_types['Dtype_Dspk'][(spk,spk2,min_idx,max_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)])*spk_samp_func(W_spk_types[(spk2,type_jdx)])
        return p_spk_types
        
    
    def generate_possibilities(self, std_descr):
        """Generate possibilities between (types,speakers) and tokens/realisations
        
        """
        pairs = {'Stype_Sspk' : {},
                 'Stype_Dspk' : {},
                 'Dtype_Sspk' : {},
                 'Dtype_Dspk' : {}}
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
                        pairs[pair_type][(speakers[tok1],types[tok1])].append((tok1, tok2))
                    if pair_type == 'Stype_Dspk':
                        pairs[pair_type][(speakers[tok1],speakers[tok2],types[tok1])].append((tok1, tok2))
                    if pair_type == 'Dtype_Sspk':
                        min_idx,max_idx = np.min([types[tok1],types[tok2]]), np.max([types[tok1],types[tok2]])
                        pairs[pair_type][(speakers[tok1],min_idx,max_idx)].append((tok1, tok2))
                    if pair_type == 'Dtype_Dspk':
                        min_idx,max_idx = np.min(types[tok1],types[tok2]), np.max([types[tok1],types[tok2]])
                        pairs[pair_type][(speakers[tok1],speakers[tok2],min_idx,max_idx)].append((tok1, tok2))
                except:
                    if pair_type == 'Stype_Sspk':
                        pairs[pair_type][(speakers[tok1],types[tok1])] = [(tok1, tok2)]
                    if pair_type == 'Stype_Dspk':
                        pairs[pair_type][(speakers[tok1],speakers[tok2],types[tok1])] = [(tok1, tok2)]
                    if pair_type == 'Dtype_Sspk':
                        min_idx,max_idx = np.min([types[tok1],types[tok2]]), np.max([types[tok1],types[tok2]])
                        pairs[pair_type][(speakers[tok1],min_idx,max_idx)] = [(tok1, tok2)]
                    if pair_type == 'Dtype_Dspk':
                        min_idx,max_idx = np.min([types[tok1],types[tok2]]), np.max([types[tok1],types[tok2]])
                        pairs[pair_type][(speakers[tok1],speakers[tok2],min_idx,max_idx)]= [(tok1, tok2)]
        return pairs
                
    
    def type_speaker_sampling_p(self, std_descr, type_samp = 'f', speaker_samp='f'):
        """Sampling proba modes for p_i1,i2,j1,j2 based on conditionnal Bayes proba:
            - log : proporitonal to log of speaker or type probabilities
            - f : proportional to square roots of speaker
                or type probabilities (in order to obtain sampling
                probas for pairs proportional to geometric mean of 
                the members of the pair probabilities)
            - f2 : proportional to speaker
                or type probabilities
            - fcube: proportionnal to log probabilities
            - 1 : equiprobable
            
        """
        assert type_samp in ['1', 'f', 'f2','log', 'fcube']
        assert speaker_samp in ['1', 'f', 'f2','log', 'fcube'] 
        #W_types = std_descr['types']
        #speakers = [e for e in std_descr['speakers']]
        #W_speakers = [std_descr['speakers'][e] for e in speakers]
        p_types = self.type_sample_p(std_descr,  type_samp = type_samp)
        p_spk_types = self.sample_spk_p(std_descr, spk_samp = speaker_samp)
        
        for config in p_types.keys():
            p_types[config] = utils.normalize_distribution(p_types[config])
        
        for config in p_spk_types.keys():
            p_spk_types[config] = utils.normalize_distribution(p_spk_types[config])
        
        for config in p_spk_types.keys():
            if config == 'Stype_Sspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Stype'][el[1]]*p_spk_types[config][el]
            if config == 'Stype_Dspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Stype'][el[2]]*p_spk_types[config][el]
            if config == 'Dtype_Sspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Dtype'][(el[1],el[2])]*p_spk_types[config][el]
            if config == 'Dtype_Dspk':
                for el in p_spk_types[config].keys():
                    p_spk_types[config][el] = p_types['Dtype'][(el[2],el[3])]*p_spk_types[config][el]
    
        for config in p_spk_types.keys():
            p_spk_types[config] = utils.normalize_distribution(p_spk_types[config])
    
        return p_spk_types
    
    def compute_cdf(self, proba):
        cdf = {}
        for key in proba.keys():
            cdf[key] = utils.cumulative_distribution(proba[key])
        return cdf

class SamplerClusterSiamese(SamplerCluster):
    """Sampler for Siamese network based on clusters of words
    
    """
    def __init__(self, type_samp='log', spk_samp='log'):
        super(SamplerClusterSiamese, self).__init__() 
        self.type_samp = type_samp
        self.spk_samp = spk_samp
        
    def whoami(self):
        return {'params':self.__dict__,'class_name': self.__class__.__name__} 
        
    def sample_batch(self,
                     p_spk_types,
                     cdf,
                     pairs,
                     seed=0, prefix='',
                     num_examples = 5012,
                     ratio_same_diff=0.25):
            
        np.random.seed(seed)
        sampled_tokens = {'Stype_Sspk' : [],
                 'Stype_Dspk' : [],
                 'Dtype_Sspk' : [],
                 'Dtype_Dspk' : []}
        num_same_spk = int((num_examples)*ratio_same_diff)
        num_diff_spk = num_examples - num_same_spk
        sampled_ratio = {
                 'Stype_Sspk' : num_same_spk/2,
                 'Stype_Dspk' : num_diff_spk/2,
                 'Dtype_Sspk' : num_same_spk/2,
                 'Dtype_Dspk' : num_diff_spk/2}
        for config in p_spk_types.keys():
            #proba_config = np.array(p_spk_types[config].values())
            #sizes = len(p_spk_types[config].keys())
            keys = np.array(p_spk_types[config].keys())
            sample_idx = utils.sample_searchidx(cdf,config,sampled_ratio[config])
            sample = keys[sample_idx]
            if config == 'Stype_Sspk':
                for key in sample:
                    spk, type_idx = key
                    pot_tok = pairs[config][spk,int(type_idx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
            if config == 'Stype_Dspk':
                for key in sample:
                    spk1,spk2, type_idx = key
                    try:
                        pot_tok = pairs[config][spk1,spk2,int(type_idx)]
                    except:
                        pot_tok = pairs[config][spk2,spk1,int(type_idx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
            if config == 'Dtype_Sspk':
                for key in sample:
                    spk, type_idx, type_jdx = key
                    pot_tok = pairs[config][spk,int(type_idx),int(type_jdx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
            if config == 'Dtype_Dspk':
                for key in sample:
                    spk1,spk2 ,type_idx, type_jdx = key
                    try:
                        pot_tok = pairs[config][spk1,spk2,int(type_idx),int(type_jdx)]
                    except:
                        pot_tok = pairs[config][spk2,spk1,int(type_idx),int(type_jdx)]
                    num_tok = len(pot_tok)
                    sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
        return sampled_tokens
    
    def write_tokens(self, descr,proba,cdf,pairs,size_batch,num_batches,out_dir,idx_batch,seed=0):
        lines = []
        np.random.seed(seed)
        sampled_batch = self.sample_batch(proba,cdf,pairs,num_pairs_batch=num_batches)
        for config in sampled_batch.keys():
            if config == 'Stype_Sspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
            if config == 'Stype_Dspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
            if config == 'Dtype_Sspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
            if config == 'Dtype_Dspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = utils.print_token(descr['tokens'][pair[0]])
                    tok2 = utils.print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
        
        np.random.shuffle(lines)
        #prev_idx = 0
        for idx in range(1,num_batches//size_batch):    
            with open(os.path.join(out_dir,'pair_'+str(idx_batch))+'_'+str(idx)+'.batch', 'w') as fh:
                    fh.writelines(lines[(idx-1)*size_batch:(idx)*size_batch])

if __name__ == '__main__':
    
    sam = SamplerClusterSiamese()
    

        
        
        
        
        
        
        
        
        
        