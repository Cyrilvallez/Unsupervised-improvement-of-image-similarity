#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:28:49 2022

@author: cyrilvallez
"""

import numpy as np
import os
import argparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from helpers import utils, plot

def find_threshold(Z, N_clusters):
    """
    Find the threshold and the cluster assignments corresponding to the given number of
    clusters desired. It uses a very simple dichotomy algorithm.

    Parameters
    ----------
    Z : Numpy array
        The linkage matrix.
    N_clusters : int
        The number of clusters desired.

    Returns
    -------
    clusters : Numpy array
        The clusters assignment corresponding to N_clusters.
    m : float
        The threshold in clustering distance corresponding to the cutting of
        the dendrogram making exactly N_clusters clusters.

    """
    
    a = 0
    b = np.max(Z[:,2])
    m = (a+b)/2
    clusters = fcluster(Z, m, criterion='distance')
    N = int(max(clusters))

    while N != N_clusters:
        
        if N < N_clusters:
            b = m
        else:
            a = m
            
        m = (a+b)/2
        clusters = fcluster(Z, m, criterion='distance')
        N = int(max(clusters))
        
    return clusters, m


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clustering of the memes')
    parser.add_argument()
    parser.add_argument('algo', type=str, default='SimCLR v2 ResNet50 2x',
                        help='The algorithm from which the features describing the images derive.')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='The metric for distance between features.')
    parser.add_argument('--linkage', type=str, default='ward', 
                        choices=['single', 'complete', 'average', 'centroid', 'ward'],
                        help='The linkage method for merging clusters.')
    args = parser.parse_args()

    algorithm = args.algo
    metric = args.metric
    linkage_type = args.linkage

    # Force usage of hamming distance for Dhash
    if 'bits' in algorithm:
        metric = 'hamming'
    
    if (metric != 'euclidean' and linkage_type == 'ward'):
        raise ValueError('Ward linkage can only be used for euclidean metric.')

    identifier = '_'.join(algorithm.split(' '))

    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    folder = f'Clustering/{identifier}_{metric}_{linkage_type}/'
    os.makedirs(folder)

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    try:  
        distances = np.load(f'Clustering/distances_{identifier}_{metric}.npy')
    except FileNotFoundError:
        distances = pdist(features, metric=metric)
        np.save(f'Clustering/distances_{identifier}_{metric}.npy', distances)
      

    # Perform hierarchical clustering
    Z = linkage(distances, method=linkage_type)
    # Maximum height of the dendrogram

    N_clusters = [500, 450, 400, 350, 300, 250, 200, 150, 100]

    rng = np.random.default_rng(seed=112)

    for i, N_cluster in enumerate(N_clusters):
    
        clusters, threshold = find_threshold(Z, N_cluster)
        
        current_dir = folder + f'{N_cluster}-clusters_thresh-{threshold:.3f}/'
        os.makedirs(current_dir, exist_ok=True)
    
        for cluster_idx in range(1, N_cluster+1):
        
            images = mapping[clusters == cluster_idx]
            if len(images) > 10:
                representatives = rng.choice(images, size=10, replace=False)
            else:
                representatives = images
            
            representation = utils.cluster_representation(representatives)
            representation.save(current_dir + f'{cluster_idx}.png')
            
    plot.plot_dendrogram(Z, linkage_type, save=True, filename=folder + 'dendrogram.pdf')
       
        

