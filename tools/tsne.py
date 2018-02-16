"""
Send an embedding to tensorboard for visualisation
"""
#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
import numpy as np
import random

from collections import defaultdict
from abnet3.utils import read_feats
import tensorboardX

from scipy.spatial.distance import cosine as cosine_distance
from scipy.special import kl_div

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

def read_item(item_path):
    # return np.genfromtxt(
    #     item_path,
    #     delimiter=' ',
    #     dtype=None,
    #     # dtype={
    #     #     'names': ['file', 'onset', 'offset', 'phone', 'context', 'talker'],
    #     #     'formats': ['S', np.float, np.float, str, str, str]},
    #     skip_header=1,
    #     # names=['file', 'onset', 'offset', 'phone', 'context', 'talker'],
    # )

    lines = []
    with open(item_path, 'r') as f:
        f.readline()
        for l in f:
            file, onset, offset, phone, context, talker = l.strip().split()
            onset, offset = float(onset), float(offset)
            lines.append([file, onset, offset, phone, context, talker])
    return lines


def group_by_phoneme(items):
    phoneme_occurences = defaultdict(list)

    for file, onset, offset, phone, context, talker in items:
        phoneme_occurences[phone].append([file, onset, offset, context, talker])
    return phoneme_occurences



def get_phoneme(file, onset, offset, features):
    frames = features.get(file, onset, offset)
    n = len(frames)
    return frames[n//2]


def get_data(features_path, item_path, phoneme_per_class=100):

    features, _, _ = read_feats(features_path)

    items = read_item(item_path)
    phoneme_occurences = group_by_phoneme(items)

    frames = []
    labels = []
    for label in phoneme_occurences:
        occurences = random.sample(phoneme_occurences[label], phoneme_per_class)
        for file, onset, offset, _, _ in occurences:
            frames.append(get_phoneme(file, onset, offset, features))
            labels.append(label)
    return np.vstack(frames), labels



if __name__ == '__main__':

    writer = tensorboardX.SummaryWriter(log_dir='./runs')

    items_path = '/home/cdancette/projects/dataset/english.item'
    features_path = "/home/cdancette/projects/dataset/embedded-15-02-frame-batchnorm.features"

    frames, labels = get_data(features_path, items_path, phoneme_per_class=5)
    lookupTable, index_labels = np.unique(labels, return_inverse=True)

    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(labels)))
    # cs = [colors[label] for label in index_labels]

    # colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colormap = plt.cm.nipy_spectral  # nipy_spectral, Set1,Paired
    colorst = [colormap(i) for i in np.linspace(0, 0.9, len(labels))]

    tsne = TSNE(metric='cosine', n_iter=5000, learning_rate=200, verbose=True, init='pca', method='exact')

    tsne.fit(frames)
    embedding = tsne.embedding_
    print(tsne.n_iter_)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=colorst)

    for i in range(len(labels)):
        plt.annotate(labels[i], (embedding[i, 0], embedding[i, 1]))

    plt.show()




    # np.savetxt('./tsne.txt', transformed_frames.embedding_)

    # writer.add_embedding(torch.from_numpy(frames), metadata=labels)
