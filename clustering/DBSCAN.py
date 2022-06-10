#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:46:36 2022

@author: cyrilvallez
"""

import numpy as np
import os
import argparse
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from helpers import utils
from clustering.clustering_utils import save_representatives, cluster_size_plot
    

def cluster_DBSCAN(algorithm, metric, min_samples):
    """
    Perform DBSCAN clustering and save cluster assignments and representative
    cluster images.

    Parameters
    ----------
    algorithm : str
        Algorithm used to extract image features.
    metric : str
        The metric used for the distance between images.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    Returns
    -------
    None.

    """

    # Force usage of hamming distance for Dhash
    if 'bits' in algorithm:
        metric = 'hamming'

    identifier = '_'.join(algorithm.split(' '))

    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    folder = f'Clustering_results/{metric}_DBSCAN_{identifier}_{min_samples}_samples/'
    os.makedirs(folder)

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    try:  
        distances = np.load(f'Clustering_results/distances_{identifier}_{metric}.npy')
    except FileNotFoundError:
        distances = pdist(features, metric=metric)
        np.save(f'Clustering_results/distances_{identifier}_{metric}.npy', distances)
        
    # Reshape the distances as a symmetric matrix
    distances = squareform(distances)
    
    if metric == 'euclidean':
        precisions = np.linspace(4, 4.5, 5)
    elif metric == 'cosine':
        precisions = np.linspace(0.18, 0.25, 5)
    elif metric == 'hamming':
        precisions = np.linspace(0.16, 0.2, 3)
    
    np.random.seed(112)
        
    for i, precision in enumerate(precisions):
    
        clustering = DBSCAN(eps=precision, metric='precomputed', algorithm='brute',
                            min_samples=min_samples, n_jobs=10)
        clusters = clustering.fit_predict(distances)
        N_clusters = len(np.unique(clusters))
        
        current_dir = folder + f'{N_clusters}-clusters_{precision:.3f}-eps/'
        os.makedirs(current_dir, exist_ok=True)
        
        np.save(current_dir + 'assignment.npy', clusters)
    
        cluster_size_plot(clusters, save=True,
                          filename=current_dir + 'cluster_balance.pdf')
        save_representatives(clusters, mapping, current_dir)
            
            
if __name__ == '__main__':
    # Parsing of command line args
    parser = argparse.ArgumentParser(description='Clustering of the memes')
    parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                        help='The algorithm from which the features describing the images derive.')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='The metric for distance between features.')
    parser.add_argument('--samples', type=int, default=5, 
                        help='The number of samples in a neighborhood for a point to be considered as a core point.')
    args = parser.parse_args()

    algorithm = ' '.join(args.algo)
    metric = args.metric
    min_samples = args.samples
    
    # Clustering
    cluster_DBSCAN(algorithm, metric, min_samples)
        
        
        
        
        