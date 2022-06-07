#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:46:36 2022

@author: cyrilvallez
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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
    ax2.set(xlabel='Number of images inside the clusters')
    ax2.set(xscale='log')

    fig.tight_layout()
    if save:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()
    

def cluster_DBSCAN():

    parser = argparse.ArgumentParser(description='Clustering of the memes')
    parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                        help='The algorithm from which the features describing the images derive.')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='The metric for distance between features.')
    args = parser.parse_args()

    algorithm = ' '.join(args.algo)
    metric = args.metric

    # Force usage of hamming distance for Dhash
    if 'bits' in algorithm:
        metric = 'hamming'

    identifier = '_'.join(algorithm.split(' '))

    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    folder = f'Clustering/{metric}_DBSCAN_{identifier}/'
    os.makedirs(folder)

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    try:  
        distances = np.load(f'Clustering/distances_{identifier}_{metric}.npy')
    except FileNotFoundError:
        distances = pdist(features, metric=metric)
        np.save(f'Clustering/distances_{identifier}_{metric}.npy', distances)
        
    # Reshape the distances as a symmetric matrix
    distances = squareform(distances)
    
    if metric == 'euclidean':
        precisions = np.linspace(4, 4.5, 5)
    elif metric == 'cosine':
        precisions = np.linspace(0.18, 0.25, 5)
    elif metric == 'hamming':
        precisions = np.linspace(0.18, 0.25, 5)
    
    rng = np.random.default_rng(seed=112)
        
    for i, precision in enumerate(precisions):
    
        clustering = DBSCAN(eps=precision, metric='precomputed', algorithm='brute',
                            n_jobs=10)
        clusters = clustering.fit_predict(distances)
        
        current_dir = folder + f'eps-{precision:.3f}/'
        os.makedirs(current_dir, exist_ok=True)
        
        cluster_size_plot(clusters, save=True,
                          filename=current_dir + 'cluster_balance.pdf')
        np.save(current_dir + 'assignment.npy', clusters)
    
        for cluster_idx in np.unique(clusters):
        
            images = mapping[clusters == cluster_idx]
            if len(images) > 10:
                representatives = rng.choice(images, size=10, replace=False)
            else:
                representatives = images
            
            representation = utils.cluster_representation(representatives)
            representation.save(current_dir + f'{cluster_idx}.png')
            
            
if __name__ == '__main__':
    cluster_DBSCAN()
        
        
        
        
        