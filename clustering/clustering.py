#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:28:49 2022

@author: cyrilvallez
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from helpers import utils
from clustering import tools

def find_threshold(Z, N_clusters, max_iter=1e4):
    """
    Find the threshold and the cluster assignments corresponding to the given number of
    clusters desired. It uses a very simple dichotomy algorithm.

    Parameters
    ----------
    Z : Numpy array
        The linkage matrix.
    N_clusters : int
        The number of clusters desired.
    max_iter : int or float, optional
        Maximum number of iterations the algorithm should perform. The default
        is 1e4.

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

    iter_count = 0
    while N != N_clusters and iter_count < max_iter:
        
        if N < N_clusters:
            b = m
        else:
            a = m
            
        m = (a+b)/2
        clusters = fcluster(Z, m, criterion='distance')
        N = int(max(clusters))
        iter_count += 1
        
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


def hierarchical_clustering(algorithm, metric, linkage_type):
    """
    Perform hierarchical clustering and save cluster assignments, representative
    cluster images, and dendrogram.
    
    Parameters
    ----------
    algorithm : str
        Algorithm used to extract image features.
    metric : str
        The metric used for the distance between images.
    linkage_type : str
        The linkage type for merging clusters.

    Raises
    ------
    ValueError
        If the combination of metric and linkage is not valid.

    Returns
    -------
    None

    """

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

    folder = f'Clustering_results/{metric}_{linkage_type}_{identifier}/'
    os.makedirs(folder)

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    try:  
        distances = np.load(f'Clustering_results/distances_{identifier}_{metric}.npy')
    except FileNotFoundError:
        distances = pdist(features, metric=metric)
        np.save(f'Clustering_results/distances_{identifier}_{metric}.npy', distances)
      

    # Perform hierarchical clustering
    Z = linkage(distances, method=linkage_type)

    # Number of clusters we desire
    N_clusters = [500, 450, 400, 350, 300, 250, 200, 150, 100]
    
    np.random.seed(112)

    for i, N_cluster in enumerate(N_clusters):
    
        clusters, threshold = find_threshold(Z, N_cluster)
        # Recompute the number of clusters, in case find_threshold did
        # not converge
        N_cluster = int(max(clusters))
        
        current_dir = folder + f'{N_cluster}-clusters_thresh-{threshold:.3f}/'
        os.makedirs(current_dir, exist_ok=True)
        
        np.save(current_dir + 'assignment.npy', clusters)
        
        tools.cluster_size_plot(clusters, save=True,
                                filename=current_dir + 'cluster_balance.pdf')
        tools.save_representatives(clusters, mapping, current_dir)
            
    plot_dendrogram(Z, linkage_type, save=True, filename=folder + 'dendrogram.pdf',
                    truncate_mode='level', p=25)
    
    
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
    
        tools.cluster_size_plot(clusters, save=True,
                                filename=current_dir + 'cluster_balance.pdf')
        tools.save_representatives(clusters, mapping, current_dir)
    

    
       
        

