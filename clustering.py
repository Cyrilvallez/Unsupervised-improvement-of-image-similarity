#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:28:49 2022

@author: cyrilvallez
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

from helpers import utils

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


def plot_dendrogram(Z, linkage, save=False, filename=None, **kwargs):
    """
    Plot the dendrogram from linkage matrix.

    Parameters
    ----------
    Z : Numpy array
        Linkage matrix.
    linkage : str
        Linkage type used.
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
    
    plt.figure(figsize=(10,10))
    dendrogram(Z, **kwargs)
    plt.xticks([])
    plt.xlabel('Image number')
    plt.ylabel('Cluster distance (linkage: ' + linkage + ')')
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
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


def hierarchical_clustering():
    """
    Perform hierarchical clustering and save representatives of each clusters.

    Raises
    ------
    ValueError
        If the combination of metric and linkage is not valid.

    Returns
    -------
    None.

    """

    parser = argparse.ArgumentParser(description='Clustering of the memes')
    parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                        help='The algorithm from which the features describing the images derive.')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='The metric for distance between features.')
    parser.add_argument('--linkage', type=str, default='ward', 
                        choices=['single', 'complete', 'average', 'centroid', 'ward'],
                        help='The linkage method for merging clusters.')
    args = parser.parse_args()

    algorithm = ' '.join(args.algo)
    metric = args.metric
    linkage_type = args.linkage

    # Force usage of hamming distance for Dhash
    if 'bits' in algorithm:
        metric = 'hamming'
    
    if (metric != 'euclidean' and linkage_type == 'ward'):
        raise ValueError('Ward linkage can only be used for euclidean metric.')
        
    if (metric != 'euclidean' and linkage_type == 'centroid'):
        raise ValueError('Centroid linkage can only be used for euclidean metric.')

    identifier = '_'.join(algorithm.split(' '))

    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    folder = f'Clustering/{metric}_{linkage_type}_{identifier}/'
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

    # Number of clusters we desire
    N_clusters = [500, 450, 400, 350, 300, 250, 200, 150, 100]

    rng = np.random.default_rng(seed=112)

    for i, N_cluster in enumerate(N_clusters):
    
        clusters, threshold = find_threshold(Z, N_cluster)
        
        current_dir = folder + f'{N_cluster}-clusters_thresh-{threshold:.3f}/'
        os.makedirs(current_dir, exist_ok=True)
        
        cluster_size_plot(clusters, save=True,
                          filename=current_dir + 'cluster_balance.pdf')
        np.save(current_dir + 'assignment.npy', clusters)
    
        for cluster_idx in range(1, N_cluster+1):
        
            images = mapping[clusters == cluster_idx]
            if len(images) > 10:
                representatives = rng.choice(images, size=10, replace=False)
            else:
                representatives = images
            
            representation = utils.cluster_representation(representatives)
            representation.save(current_dir + f'{cluster_idx}.png')
            
    plot_dendrogram(Z, linkage_type, save=True, filename=folder + 'dendrogram.pdf',
                    truncate_mode='level', p=25)
    
            
if __name__ == '__main__':
    hierarchical_clustering()
       
        

