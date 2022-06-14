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


def save_representatives(clusters, mapping, current_dir):
    """
    Save representatives of each clusters from the cluster assignments.

    Parameters
    ----------
    directory : str
        Path to the results.
    mapping : Numpy array
        Mapping from indices to image paths.

    Returns
    -------
    None.

    """
    
    for cluster_idx in np.unique(clusters):
    
        images = mapping[clusters == cluster_idx]
        if len(images) > 10:
            representatives = np.random.choice(images, size=10, replace=False)
        else:
            representatives = images
        
        representation = cluster_representation(representatives)
        representation.save(current_dir + f'{cluster_idx}.png')
        
        
def get_cluster_diameters(assignments, algorithm, metric):
    """
    Compute the diameter of each clusters.

    Parameters
    ----------
    assignments : Numpy array
        The clusters assignments.
    algorithm : str
        The algorithm used to extract features.
    metric : str
        The metric used for the distances.

    Returns
    -------
    Numpy array
        The diameter of each cluster.

    """
    identifier = '_'.join(algorithm.split(' '))

    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    
    try:  
        distances = np.load(f'Clustering_results/distances_{identifier}_{metric}.npy')
    except FileNotFoundError:
        distances = pdist(features, metric=metric)
        np.save(f'Clustering_results/distances_{identifier}_{metric}.npy', distances)
        
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


def get_cluster_centroids(assignment, algorithm, metric):
    """
    Compute the centroids of each clusters.

    Parameters
    ----------
    assignments : Numpy array
        The clusters assignments.
    algorithm : str
        The algorithm used to extract features.
    metric : str
        The metric used for the distances.

    Returns
    -------
    Numpy array
        The diameter of each cluster.

    """

    assignment = np.load(folder)
    unique, counts = np.unique(assignment, return_counts=True)

    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)

    engine = NearestCentroid(metric=metric)
    engine.fit(features, assignment)
    
    # They are already sorted correctly with respect to the cluster indices
    return engine.centroids_



def save_diameters(experiment_folder):
    """
    Save the diameter of each cluster to file for later reuse.

    Parameters
    ----------
    experiment_folder : str
        Path to the folder containing the clustering results.

    Returns
    -------
    None.

    """
    
    algorithm = experiment_folder.rsplit('/', 2)[1].split('_', 2)[-1]
    algorithm = ' '.join(algorithm.split('_'))
    if 'samples' in algorithm:
        algorithm = algorithm.rsplit(' ', 2)[0]
        
    metric = experiment_folder.rsplit('/', 2)[1].split('_', 1)[0]
    
    for subfolder in tqdm([f.path for f in os.scandir(experiment_folder) if f.is_dir()]):
        
        clusters = np.load(subfolder + '/assignment.npy')
        diameters = get_cluster_diameters(clusters, algorithm, metric)
        np.save(subfolder + '/diameters.npy', diameters)
        
        
        
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
    
    
def cluster_size_diameter_plot(assignments, diameters, metric, save=False, filename=None):
    """
    Creates a scatter plot showing the diameter of each cluster against its size.

    Parameters
    ----------
    assignments : Numpy array
        The clusters assignments.
    diameters : Numpy array
        The diameters of each cluster.
    metric : str
        The metric used for the distances.
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
        
    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'
    
    algorithm = directory.rsplit('/', 2)[1].split('_', 2)[-1]
    algorithm = ' '.join(algorithm.split('_'))
    if 'samples' in algorithm:
        algorithm = algorithm.rsplit(' ', 2)[0]

    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
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
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
            
    if directory[-1] != '/':
        directory += '/'
            
    diameters = []
        
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
    
    algorithm = directory.rsplit('/', 2)[1].split('_', 2)[-1]
    algorithm = ' '.join(algorithm.split('_'))
    if 'samples' in algorithm:
        algorithm = algorithm.rsplit(' ', 2)[0]
        
    metric = directory.rsplit('/', 2)[1].split('_', 1)[0]
        
    for folder in subfolders:
        diameters.append(np.load(folder + '/diameters.npy'))
    
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
    
    
    
    
        
if __name__ == '__main__':

    # directory = 'Clustering_results'
    # for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
        # counts = cluster_size_violin(folder, save=True, filename='sizes_violin.pdf')
        # cluster_size_evolution(folder, save=False, filename='sizes.pdf')
        # cluster_diameter_violin(folder, save=True, filename='sizes_violin.pdf')
        
    directory = 'Clustering_results'
    for folder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        save_diameters(folder)
    
"""
#%%
def clusters_intersection(assignment1, assignment2, algorithm):
    
    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)

    cluster_indices1 = np.unique(assignment1)
    cluster_indices2 = np.unique(assignment2)
    
    intersection_percentage = np.empty((len(cluster_indices1), len(cluster_indices2)))
    
    for i, index1 in enumerate(cluster_indices1):
        
        cluster1 = np.argwhere(assignment1 == index1).flatten()

        for j, index2 in enumerate(cluster_indices2):
            
            cluster2 = np.argwhere(assignment2 == index2).flatten()
            intersection_percentage[i,j] = len(np.intersect1d(cluster1, cluster2,
                                                              assume_unique=True))/len(cluster1)

    return intersection_percentage, cluster_indices1, cluster_indices2


def intersection_plot(folder1, folder2, save=False, filename=None):

    if folder1[-1] != '/':
        folder1 += '/'
    if folder2[-1] != '/':
        folder2 += '/'
        
    assert (folder1.rsplit('/', 2)[0] == folder2.rsplit('/', 2)[0])
    
    algorithm = folder1.rsplit('/', 3)[1].split('_', 2)[-1]
    algorithm = ' '.join(algorithm.split('_'))
    if 'samples' in algorithm:
        algorithm = algorithm.rsplit(' ', 2)[0]
        
    assignment1 = np.load(folder1 + 'assignment.npy')
    assignment2 = np.load(folder2 + 'assignment.npy')
    intersection, indices1, indices2 = clusters_intersection(assignment1, assignment2,
                                                             algorithm)
    
    mask = intersection == 0.

    plt.figure(figsize=(20,20))
    sns.heatmap(intersection, annot=False, fmt='.2%', xticklabels=indices2,
                yticklabels=indices1)
    if save:
        plt.savefig(filename, bbox_inches='tight')

    return intersection, indices1, indices2
        
#%%

algorithm = 'SimCLR v2 ResNet50 2x'
metric = 'cosine'
folder = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/268-clusters_4.375-eps'
folder2 = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/306-clusters_4.250-eps'
# assignment1 = np.load(folder + '/assignment.npy')
# assignment2 = np.load(folder2 + '/assignment.npy')
intersection, indices1, indices2 = intersection_plot(folder, folder2, save=True, filename='test.pdf')

indices2 = indices2[0:len(indices1)]
intersection = intersection[0:len(indices1), 0:len(indices1)]
plt.figure(figsize=(8,8))
plt.hexbin(indices2, indices1, intersection)
plt.savefig('test_hexbin.pdf')
        
"""      