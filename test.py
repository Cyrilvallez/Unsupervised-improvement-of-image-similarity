#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
import time

import experiment as ex
from helpers import utils
"""
algorithm = 'Dhash 64 bits'
main_dataset = 'BSDS500_original'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'

features_db, mapping_db = utils.combine_features(algorithm, main_dataset, distractor_dataset)
features_query, mapping_query = utils.load_features(algorithm, query_dataset)

res = faiss.StandardGpuResources()

index = faiss.GpuIndexBinaryFlat(res, features_db.shape[1]*8)
# index = faiss.index_cpu_to_all_gpus(index)
# index = faiss.index_cpu_to_gpu(res, 0, index)

t0 = time.time()

index.add(features_db)
D, I = index.search(features_query, 10)

dt = time.time() - t0

recall, _ = utils.recall(I, mapping_db, mapping_query)

print(f'{dt:.2f} s')
print(f'Recall : {recall:.2f}')
"""

"""
import torch
import torch.nn.functional as F
from helpers import utils

algorithm = 'SimCLR v2 ResNet50 2x'
dataset1 = 'Kaggle_memes'
dataset2 = 'Kaggle_templates'

features, _ = utils.combine_features(algorithm, dataset1, dataset2)
features = torch.tensor(features).to('cuda')

distances = F.pdist(features)
distances = distances.cpu().numpy()
np.save('distances_all_memes_L2', distances)
"""

# algorithm = 'SimCLR v2 ResNet50 2x'
algorithm = 'Dhash 64 bits'
main_dataset = 'BSDS500_original'
# main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'
# query_dataset = 'Kaggle_memes'

experiment = ex.Experiment(algorithm, main_dataset, query_dataset,
                        distractor_dataset=distractor_dataset)

print(experiment.binary)

#%%

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from helpers import utils

def cluster_size_plot(cluster_assignments, save=False, filename=None):
    """
    Creates a bar plot and box plot showing how much images belong to each cluster.

    Parameters
    ----------
    cluster_assignments : Numpy array
        The clusters assignments.
    save : bool, optional
        Whether to save the figure or not. The default is False.
    filename : str, optional
        Filename for saving the figure. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is not provided.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
    
    unique, counts = np.unique(cluster_assignments, return_counts=True)

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(15,7), gridspec_kw={'height_ratios': [3, 1]})

    sns.countplot(x=cluster_assignments, ax=ax1)
    ax1.set(xlabel='Cluster number')
    ax1.set(ylabel='Number of images in the cluster')
    ax1.set(yscale='log')
    ticks = ax1.get_xticks()
    ax1.set_xticks(ticks[0::10])

    sns.boxplot(x=counts, color='royalblue', ax=ax2)
    ax2.set(xlabel='Number of images inside a cluster')
    ax2.set(xscale='log')

    fig.tight_layout()
    if save:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()
    

algorithm = 'Dhash 64 bits'
# algorithm = 'SimCLR v2 ResNet50 2x'
metric = 'euclidean'

# Force usage of hamming distance for Dhash
if 'bits' in algorithm:
    metric = 'hamming'

identifier = '_'.join(algorithm.split(' '))

dataset1 = 'Kaggle_memes'
dataset2 = 'Kaggle_templates'

# Load features and mapping to actual images
# features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                       # to_bytes=False)
distances = np.load(f'Clustering/distances_{identifier}_{metric}.npy')

    
# Reshape the distances as a symmetric matrix
distances = squareform(distances)

# precisions = [3, 4, 5, 6, 7]
precisions = np.linspace(0.16, 0.2, 5)
# precisions = np.linspace(4, 4.5, 5)
sizes_hamming = []

rng = np.random.default_rng(seed=112)
    
for i, precision in enumerate(precisions):

    clustering = DBSCAN(eps=precision, metric='precomputed', algorithm='brute',
                        n_jobs=10)
    clusters = clustering.fit_predict(distances)
    sizes_hamming.append(len(np.unique(clusters)))
    
    cluster_size_plot(clusters, save=False)
"""
    np.save(current_dir + 'assignment.npy', clusters)

    for cluster_idx in np.unique(clusters):
    
        images = mapping[clusters == cluster_idx]
        if len(images) > 10:
            representatives = rng.choice(images, size=10, replace=False)
        else:
            representatives = images
        
        representation = utils.cluster_representation(representatives)
        representation.save(current_dir + f'{cluster_idx}.png')
"""




