#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:24:02 2022

@author: cyrilvallez
"""

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        
if __name__ == '__main__':
    
    directory = 'Clustering_results'
    for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
        cluster_size_evolution(folder, save=True, filename='sizes.pdf')
    
    
#%%

if directory[-1] != '/':
    directory += '/'
    
counts = []

subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
subfolders = sorted(subfolders, reverse=True,
                    key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))

assignment = np.load(subfolders[0] + '/assignment.npy')
_, count = np.unique(assignment, return_counts=True)
    
plt.figure(figsize=(15, 4))

# Sort in decreasing order
count = -np.sort(-count)
plt.bar(np.arange(len(count)), count)
plt.ylabel(f'{len(count)} clusters')
plt.xlabel('Number of cluster')
plt.yscale('log')
lims = plt.ylim()
    
    
    
    
    
    
    


