#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:24:02 2022

@author: cyrilvallez
"""

from PIL import Image
import numpy as np
import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

from helpers import utils

# =============================================================================
# In this file we refer to `directory` as the root directory of a clusterting
# experiment, e.g "Clustering_results/euclidean_ward_SimCLR_v2_ResNet50_2x", and
# to `subfolder` as a subfolder of a `directory`, e.g
# "Clustering_results/euclidean_ward_SimCLR_v2_ResNet50_2x/100-clusters_thresh-66.637"
# =============================================================================

def cluster_representation(images):
    """
    Concatenate images from the same cluster into one image to get a visual idea
    of the clustering.

    Parameters
    ----------
    images : list of image paths
        The representatives images from the cluster.

    Returns
    -------
    PIL image
        The concatenation.

    """
    
    images = [Image.open(image).convert('RGB') for image in images]
    # Resize all images to 300x300
    images = [image.resize((300,300), Image.BICUBIC) for image in images]
    
    Nlines = len(images) // 3 + 1
    Ncols = 3 if len(images) >= 3 else len(images)

    offset = 10
    
    final_image = np.zeros((300*Nlines + (Nlines-1)*offset,
                            300*Ncols + (Ncols-1)*offset, 3), dtype='uint8')
    
    start_i = 0
    start_j = 0
    index = 0
    for i in range(Nlines):
        for j in range(Ncols):
            if index < len(images):
                final_image[start_i:start_i+300, start_j:start_j+300, :] = np.array(images[index])
                index += 1
                start_j += 300 + offset
        start_j = 0
        start_i += 300 + offset
            
    return Image.fromarray(final_image)


def save_representatives(assignments, mapping, current_dir):
    """
    Save representatives of each clusters from the cluster assignments.

    Parameters
    ----------
    assignments : Numpy array
        The clusters assignments.
    mapping : Numpy array
        Mapping from indices to image paths.
    current_dir : str
        Directory where to save the images.

    Returns
    -------
    None.

    """
    
    for cluster_idx in np.unique(assignments):
    
        images = mapping[assignments == cluster_idx]
        if len(images) > 10:
            representatives = np.random.choice(images, size=10, replace=False)
        else:
            representatives = images
        
        representation = cluster_representation(representatives)
        representation.save(current_dir + f'{cluster_idx}.png')
        
        
def _is_directory(directory):
    """
    Check if `directory` is indeed the root directory of a clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Raises
    ------
    ValueError
        If this is not the case.

    Returns
    -------
    None.

    """
    
    if directory[-1] == '/':
        directory = directory.rsplit('/', 1)[0]
        
    if os.path.dirname(directory) != 'Clustering_results':
        raise ValueError('The directory you provided is not valid.')
        
        
def _is_subfolder(subfolder):
    """
    Check if `subfolder` is indeed a subfolder of a directory containing a
    clusterting experiment.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...

    Raises
    ------
    ValueError
        If this is not the case.

    Returns
    -------
    None.

    """
    
    if subfolder[-1] == '/':
        subfolder = subfolder.rsplit('/', 1)[0]
        
    split = subfolder.rsplit('/', 2)
    if split[0] != 'Clustering_results' or len(split) != 3:
        raise ValueError('The subfolder you provided is not valid.')
        
        
def extract_params_from_folder_name(directory):
    """
    Extract the algorithm name and metric used from the directory name of
    a clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained, or subfolder of this
        directory.

    Returns
    -------
    algorithm : str
        Algorithm name used for the experiment.
    metric : str
        Metric used for the experiment.

    """
    
    if directory[-1] == '/':
        directory = directory.rsplit('/', 1)[0]
        
    # If True, this is a child folder of the experiment directory
    if 'clusters' in directory.rsplit('/', 1)[1]:
        directory = os.path.dirname(directory)
    
    algorithm = directory.rsplit('/', 1)[1].split('_', 2)[-1]
    algorithm = ' '.join(algorithm.split('_'))
    if 'samples' in algorithm:
        algorithm = algorithm.rsplit(' ', 2)[0]
        
    metric = directory.rsplit('/', 2)[1].split('_', 1)[0]
    
    return algorithm, metric


def extract_features_from_folder_name(directory, return_distances=False):
    """
    Extract the features and mapping to images from the directory name of
    a clustering experiment.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained, or subfolder of this
        directory.

    Returns
    -------
    features : Numpy array
        The features corresponding to the images.
    mapping : Numpy array
        Mapping from feature index to actual image (as a path name).
    return_distances : bool, optional
        If `True`, will also return the distances between each features. The 
        default is False.

    """
    
    algorithm, metric = extract_params_from_folder_name(directory)
    
    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    
    if return_distances:
        identifier = '_'.join(algorithm.split(' '))
        try:  
            distances = np.load(f'Clustering_results/distances_{identifier}_{metric}.npy')
        except FileNotFoundError:
            distances = pdist(features, metric=metric)
            np.save(f'Clustering_results/distances_{identifier}_{metric}.npy', distances)
        
        return features, mapping, distances
    
    else:
        return features, mapping   
 
        
def compute_cluster_diameters(subfolder):
    """
    Compute the diameter of each clusters.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...

    Returns
    -------
    Numpy array
        The diameter of each cluster.

    """
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    features, _, distances = extract_features_from_folder_name(subfolder,
                                                                     return_distances=True)
    assignments = np.load(subfolder + 'assignment.npy')
        
    # Mapping from distance matrix indices to condensed representation index
    N = len(features)
    
    def square_to_condensed(i, j):
        assert i != j, "no diagonal elements in condensed matrix"
        if i < j:
            i, j = j, i
        return N*j - j*(j+1)//2 + i - 1 - j
        
    diameters = []
    for cluster_idx in np.unique(assignments):
        
        correct_indices = np.argwhere(assignments == cluster_idx).flatten()
        
        condensed_indices = [square_to_condensed(i,j) for i,j \
                             in itertools.combinations(correct_indices, 2)]
        cluster_distances = distances[condensed_indices]
        
        # If the cluster contains only 1 image
        if len(cluster_distances) == 0:
            diameters.append(0.)
        else:
            diameters.append(np.max(cluster_distances))
        
    return np.array(diameters)


def compute_cluster_centroids(subfolder):
    """
    Compute the centroids of each clusters.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...

    Returns
    -------
    Numpy array
        The diameter of each cluster.

    """
    
    _is_subfolder(subfolder)

    if subfolder[-1] != '/':
        subfolder += '/'
    
    _, metric = extract_params_from_folder_name(subfolder)
    features, _ = extract_features_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
    
    unique, counts = np.unique(assignments, return_counts=True)

    engine = NearestCentroid(metric=metric)
    engine.fit(features, assignments)
    
    # They are already sorted correctly with respect to the cluster indices
    return engine.centroids_


def _get_attribute(subfolder, func, identifier, save_if_absent=True):
    """
    Return the attribute computed by `func`, trying to load it from disk,
    then computing it if it is absent.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...
    func : function
        The function computing the attribute.
    identifier : str
        String representing the attribute. E.g `diameters`, `centroids`.
    save_if_absent : bool, optional
        Whether or not to save the result is they are not already on disk. The
        default is True.

    Returns
    -------
    diameters : Numpy array
        The diameters.

    """
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    try:
        attribute = np.load(subfolder + identifier + '.npy')
    except FileNotFoundError:
        attribute = func(subfolder)
        if save_if_absent:
            np.save(subfolder + identifier + '.npy', attribute)
            
    return attribute


def get_cluster_diameters(subfolder, save_if_absent=True):
    """
    Return the cluster diameters, trying to load them from disk then computing
    them otherwise.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...
    save_if_absent : bool, optional
        Whether or not to save the result is they are not already on disk. The
        default is True.

    Returns
    -------
    centroids : Numpy array
        The diameters.

    """
    
    return _get_attribute(subfolder, compute_cluster_diameters, 'diameters',
                          save_if_absent=save_if_absent)


def get_cluster_centroids(subfolder, save_if_absent=True):
    """
    Return the cluster centroids, trying to load them from disk then computing
    them otherwise.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...
    save_if_absent : bool, optional
        Whether or not to save the result is they are not already on disk. The
        default is True.

    Returns
    -------
    centroids : Numpy array
        The diameters.

    """
    
    return _get_attribute(subfolder, compute_cluster_centroids, 'centroids',
                          save_if_absent=save_if_absent)


def _save_attribute(directory, func, identifier):
    """
    Save the attribute computed by `func` to disk, for each subfolder of
    `directory`.
    
    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.
    func : function
        The function computing the attribute.
    identifier : str
        String representing the attribute. E.g `diameters`, `centroids`.

    Returns
    -------
    None

    """
    
    _is_directory(directory)
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        
        attribute = func(subfolder)
        np.save(subfolder + identifier + '.npy', attribute)
        

def save_diameters(directory):
    """
    Save the diameter of each cluster to file for later reuse.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Returns
    -------
    None.

    """
    
    _save_attribute(directory, compute_cluster_diameters, 'diameters')
        

def save_centroids(directory):
    """
    Save the diameter of each cluster to file for later reuse.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.

    Returns
    -------
    None.

    """
    
    _save_attribute(directory, compute_cluster_centroids, 'centroids')
             
        
def cluster_size_plot(assignments, save=False, filename=None):
    """
    Creates a bar plot and box plot showing how much images belong to each cluster.

    Parameters
    ----------
    assignments : Numpy array
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
    
    unique, counts = np.unique(assignments, return_counts=True)

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(15,7),
                                  gridspec_kw={'height_ratios': [3, 1]})

    sns.countplot(x=assignments, ax=ax1)
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
    
    
def cluster_size_diameter_plot(subfolder, save=False, filename=None):
    """
    Creates a scatter plot showing the diameter of each cluster against its size.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...
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
    
    _is_subfolder(subfolder)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
        
    if subfolder[-1] != '/':
        subfolder += '/'
        
    _, metric = extract_params_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
    diameters = get_cluster_diameters(subfolder)
    
    _, counts = np.unique(assignments, return_counts=True)

    plt.figure()
    plt.scatter(counts, diameters)
    plt.xscale('log')
    plt.xlabel('Cluster size')
    plt.ylabel(f'Cluster diameter ({metric} distance)')
    plt.grid()

    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
def cluster_size_evolution(directory, save=False, filename=None):
    """
    Plot the cluster sizes along with the distribution of each real clusters.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.
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
    
    _is_directory(directory)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
        
    if directory[-1] != '/':
        directory += '/'
        
    counts = []
    
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
    
    for folder in subfolders:
        assignment = np.load(folder + '/assignment.npy')
        _, count = np.unique(assignment, return_counts=True)
        counts.append(count)
        
    N = len(counts)
    fig, axes = plt.subplots(N+1, 1, figsize=(15, 15), sharex=True, sharey=True)
    
    for count, ax in zip(counts, axes[0:-1]):
        
        # Sort in decreasing order
        count = -np.sort(-count)
        ax.bar(np.arange(len(count)), count)
        ax.set(ylabel=f'{len(count)} clusters')
        ax.set(xlabel='Number of cluster')
        ax.set(yscale='log')
        
    features, mapping = extract_features_from_folder_name(directory)
    
    # Find the count of memes inside each "real" clusters (from the groundtruths)
    identifiers = []
    for name in mapping:
        identifier = name.rsplit('/', 1)[1].split('_', 1)[0]
        if '.' in identifier:
            identifier = name.rsplit('/', 1)[1].rsplit('.', 1)[0]
        identifiers.append(identifier)
        
    _, count = np.unique(identifiers, return_counts=True)
    count = -np.sort(-count)
    
    
    axes[-1].bar(np.arange(len(count)), count, color='r')
    axes[-1].set(ylabel='Original clusters')
    axes[-1].set(xlabel='Number of cluster')
    axes[-1].set(yscale='log')
    lims = axes[-1].get_ylim()
    axes[-1].set_ylim(0.64, lims[1])
    
    if save:
        fig.savefig(directory + filename, bbox_inches='tight')
    plt.show()
    
    
def cluster_size_violin(directory, cut=2, save=False, filename=None):
    """
    Plot the cluster sizes as a violin plot.

    Parameters
    ----------
    directory : str
        Directory where the results are.
    cut : float, optional 
        Distance, in units of bandwidth size, to extend the density past the
        extreme datapoints. Set to 0 to limit the violin range within the
        range of the observed data. The default is 2.
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
    
    _is_directory(directory)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
            
    if directory[-1] != '/':
        directory += '/'
            
    counts = []
        
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
        
    for folder in subfolders:
        assignment = np.load(folder + '/assignment.npy')
        _, count = np.unique(assignment, return_counts=True)
        counts.append(count)
    
    # Creates a dataframe for easy violinplot with seaborn
    order = sorted([len(counts[i]) for i in range(len(counts))], reverse=True)
    N_clusters = [len(counts[i])*np.ones(len(counts[i]), dtype=int) \
                  for i in range(len(counts))]
    N_clusters = np.concatenate(N_clusters)
    # Put data in log because violinplot cannot use log-scale directly
    counts = np.log10(np.concatenate(counts))
    
    frame = pd.DataFrame({'Number of clusters': N_clusters, 'Cluster size': counts})
    
    plt.figure(figsize=(8,8))
    sns.violinplot(x='Number of clusters', y='Cluster size', data=frame,
                   order=order, cut=cut)
    # Set the ticks correctly for log plot in a linear scale (since this is the data
    # which is in log)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    tick_range = np.arange(np.floor(ymin), ymax)
    minor_ticks = [np.log10(x) for a in tick_range \
                   for x in np.linspace(10**a, 10**(a+1), 10)[1:-1]]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tick_range))
    ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(minor_ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('$10^{{{x:.0f}}}$'))

    if save:
        plt.savefig(directory + filename, bbox_inches='tight')
    plt.show()
    
    return counts
    
    
def cluster_diameter_violin(directory, save=False, filename=None):
    """
    Plot the cluster diameters as a violin plot.

    Parameters
    ----------
    directory : str
        Directory where the results are.
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
    
    _is_directory(directory)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
            
    if directory[-1] != '/':
        directory += '/'
            
    diameters = []
        
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
    
    _, metric = extract_params_from_folder_name(directory)
        
    for folder in subfolders:
        diameters.append(get_cluster_diameters(folder))
    
    # Creates a dataframe for easy violinplot with seaborn
    order = sorted([len(diameters[i]) for i in range(len(diameters))], reverse=True)
    N_clusters = [len(diameters[i])*np.ones(len(diameters[i]), dtype=int) \
                  for i in range(len(diameters))]
    N_clusters = np.concatenate(N_clusters)
    
    frame = pd.DataFrame({'Number of clusters': N_clusters, 'Cluster diameter': diameters})
    
    plt.figure(figsize=(8,8))
    sns.violinplot(x='Number of clusters', y='Cluster diameter', data=frame,
                   order=order)
    plt.ylabel(f'Cluster diameter ({metric} distance)')

    if save:
        plt.savefig(directory + filename, bbox_inches='tight')
    plt.show()
    



    