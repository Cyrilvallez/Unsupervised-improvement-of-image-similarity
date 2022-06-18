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
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from clustering import tools
from clustering import clustering_plot as cplot

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
    folder : str
        The folder where the results have been saved.

    """

    # Force usage of hamming distance for Dhash
    if 'bits' in algorithm:
        metric = 'hamming'
    
    if (metric != 'euclidean' and linkage_type == 'ward'):
        raise ValueError('Ward linkage can only be used for euclidean metric.')
        
    if (metric != 'euclidean' and linkage_type == 'centroid'):
        raise ValueError('Centroid linkage can only be used for euclidean metric.')

    identifier = '_'.join(algorithm.split(' '))

    folder = f'Clustering_results/{metric}_{linkage_type}_{identifier}/'
    os.makedirs(folder)

    features, mapping, distances = tools.extract_features_from_folder_name(folder,
                                                                           return_distances=True)
      
    # Perform hierarchical clustering
    Z = linkage(distances, method=linkage_type)

    # Number of clusters we desire
    N_clusters = [400, 350, 300, 250, 200, 150, 100]
    
    np.random.seed(112)

    for i, N_cluster in enumerate(N_clusters):
    
        clusters, threshold = find_threshold(Z, N_cluster)
        # Recompute the number of clusters, in case find_threshold did
        # not converge
        N_cluster = int(max(clusters))
        
        current_dir = folder + f'{N_cluster}-clusters_thresh-{threshold:.3f}/'
        os.makedirs(current_dir, exist_ok=True)
        
        np.save(current_dir + 'assignment.npy', clusters)
            
    plot_dendrogram(Z, linkage_type, save=True, filename=folder + 'dendrogram.pdf',
                    truncate_mode='level', p=25)
    
    return folder
    
    
def cluster_DBSCAN(algorithm, metric, min_samples, precisions=None):
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
    precisions : list or float, optional
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. This is not a maximum bound on the
        distances of points within a cluster. If not specified, some default 
        will be set. The default is None.

    Returns
    -------
    folder : str
        The folder where the results have been saved.

    """
    
    if type(precisions) == float or type(precisions) == int:
        precisions = [precisions]

    # Force usage of hamming distance for Dhash
    if 'bits' in algorithm:
        metric = 'hamming'

    identifier = '_'.join(algorithm.split(' '))

    folder = f'Clustering_results/{metric}_DBSCAN_{identifier}_{min_samples}_samples/'
    os.makedirs(folder)

    features, mapping, distances = tools.extract_features_from_folder_name(folder,
                                                                           return_distances=True)
        
    # Reshape the distances as a symmetric matrix
    distances = squareform(distances)
    
    if precisions is None:
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
        
    return folder


def save_attributes(directory):
    """
    Save diameters and centroids for the clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Returns
    -------
    None.

    """
    
    print('Saving diameters', flush=True)
    tools.save_diameters(directory)
    print('Saving centroids', flush=True)
    tools.save_centroids(directory)
    
    
def save_plots(directory):
    """
    Save all plots for the clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Returns
    -------
    None.

    """
    
    print('Saving the different plots and image representatives', flush=True)
    cplot.cluster_size_evolution(directory, save=True, filename='sizes.pdf')
    cplot.cluster_size_violin(directory, save=True, filename='sizes_violin.pdf')
    cplot.cluster_diameter_violin(directory, save=True, filename='diameters_violin.pdf')
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        cplot.cluster_size_plot(subfolder, save=True, filename='cluster_balance.pdf')
        cplot.cluster_size_diameter_plot(subfolder, save=True, filename='size_diameter.pdf')
        
        
def save_visualizations(directory):
    """
    Save the cluster visualizations for the clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Returns
    -------
    None.

    """
    
    print('Saving visualizations')
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        tools.save_representatives(subfolder)
        tools.save_extremes(subfolder)
        

def save_all(directory):
    """
    Save everything for the clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Returns
    -------
    None.

    """
    
    save_attributes(directory)
    save_visualizations(directory)
    save_plots(directory)
       
        

