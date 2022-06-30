#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:41:01 2022

@author: cyrilvallez
"""

import matplotlib
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from clustering import tools
from helpers import plot_config


def cluster_size_plot(subfolder, save=False, filename=None):
    """
    Creates a bar plot and box plot showing how much images belong to each cluster.

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
    
    tools._is_subfolder(subfolder)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
        
    if subfolder[-1] != '/':
        subfolder += '/'
        
    assignments = np.load(subfolder + 'assignment.npy')
    
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
        fig.savefig(subfolder + filename, bbox_inches='tight')
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
    
    tools._is_subfolder(subfolder)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
        
    if subfolder[-1] != '/':
        subfolder += '/'
        
    _, metric = tools.extract_params_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
    diameters = tools.get_cluster_diameters(subfolder)
    
    _, counts = np.unique(assignments, return_counts=True)

    plt.figure()
    plt.scatter(counts, diameters)
    plt.xscale('log')
    plt.xlabel('Cluster size')
    plt.ylabel(f'Cluster diameter ({metric} distance)')
    plt.grid()

    if save:
        plt.savefig(subfolder + filename, bbox_inches='tight')
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
    
    tools._is_directory(directory)
    
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
    
    features, mapping = tools.extract_features_from_folder_name(directory)
    groundtruth_assignment = tools.get_groundtruth_attribute(directory, 'assignment')
    _, count = np.unique(groundtruth_assignment, return_counts=True)
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
    
    tools._is_directory(directory)
    
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
        
    # Add the groundtruth distribution to the plot
    groundtruth_assignment = tools.get_groundtruth_attribute(directory, 'assignment')
    _, count = np.unique(groundtruth_assignment, return_counts=True)
    counts.append(count)
    
    order = [str(len(counts[i])) for i in range(len(counts)-1)] + \
        [f'{len(counts[-1])} (original)']
    order = sorted(order)
    # Creates a dataframe for easy violinplot with seaborn
    N_clusters = [len(counts[i])*np.ones(len(counts[i]), dtype=int) \
                   for i in range(len(counts)-1)]
    N_clusters.append([f'{len(counts[-1])} (original)' for i in range(len(counts[-1]))])
    N_clusters = np.concatenate(N_clusters)
    # Put data in log because violinplot cannot use log-scale directly
    counts = np.log10(np.concatenate(counts))
    
    frame = pd.DataFrame({'Number of clusters': N_clusters, 'Cluster size': counts})
    palette = {N: 'tab:red' if 'original' in N else 'tab:blue' for N in order}
    
    plt.figure(figsize=(8,8))
    sns.violinplot(x='Number of clusters', y='Cluster size', data=frame,
                    order=order, cut=cut, palette=palette)
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
    
    locs, labels = plt.xticks()
    labels = [label.get_text().split()[0] for label in labels]
    plt.xticks(locs, labels)
    legend_elements = [Patch(facecolor='tab:blue', label='Clustering'),
                       Patch(facecolor='tab:red', label='Original')]
    plt.legend(handles=legend_elements, loc='best')

    if save:
        plt.savefig(directory + filename, bbox_inches='tight')
    plt.show()
    
    
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
    
    tools._is_directory(directory)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
            
    if directory[-1] != '/':
        directory += '/'
            
    diameters = []
        
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
        
    for folder in subfolders:
        # For DBSCAN, do not take the first diameter (which is the cluster of outliers)
        if 'DBSCAN' in directory:
            diameters.append(tools.get_cluster_diameters(folder)[1:])
        else:
            diameters.append(tools.get_cluster_diameters(folder))
        
    groundtruth_diameters = tools.get_groundtruth_attribute(directory, 'diameters')
    diameters.append(groundtruth_diameters)
    
    order = [str(len(diameters[i])) for i in range(len(diameters)-1)] + \
        [f'{len(diameters[-1])} (original)']
    order = sorted(order)
    # Creates a dataframe for easy violinplot with seaborn
    N_clusters = [len(diameters[i])*np.ones(len(diameters[i]), dtype=int) \
                   for i in range(len(diameters)-1)]
    N_clusters.append([f'{len(diameters[-1])} (original)' for i in range(len(diameters[-1]))])
    N_clusters = np.concatenate(N_clusters)
    diameters = np.concatenate(diameters)
    
    palette = {N: 'tab:red' if 'original' in N else 'tab:blue' for N in order}
    frame = pd.DataFrame({'Number of clusters': N_clusters, 'Cluster diameter': diameters})
    
    plt.figure(figsize=(8,8))
    sns.violinplot(x='Number of clusters', y='Cluster diameter', data=frame,
                   order=order, palette=palette)
    _, metric = tools.extract_params_from_folder_name(directory)
    plt.ylabel(f'Cluster diameter ({metric} distance)')
    
    locs, labels = plt.xticks()
    labels = [label.get_text().split()[0] for label in labels]
    plt.xticks(locs, labels)
    legend_elements = [Patch(facecolor='tab:blue', label='Clustering'),
                       Patch(facecolor='tab:red', label='Original')]
    plt.legend(handles=legend_elements, loc='best')

    if save:
        plt.savefig(directory + filename, bbox_inches='tight')
    plt.show()
        
    
def _reorder(a):
    """
    Reorder an array for large matrix plotting. The rows are ordered by largest
    column value in descending order. The columns are reordered such that the 
    biggest value of each row is on the main diagonal, and other values are 
    in descending order 'centered' along this diagonal value. Note that 
    columns are reordered differently on each row, thus the ordering of
    the resulting matrix does not make sense anymore and is only useful for
    plotting.
    
    Example
    --------
    
    a = np.array(
        [[1, 2, 3, 4],
        [9, 4, 4, 0],
        1, 1, 0, 2]]        
        )
        
    _reorder(a) = np.array(
        [[9, 4, 4, 0]
        [3, 4, 2, 1],
        [0, 1, 2, 1]]
        )
                  
    Parameters
    ----------
    a : Numpy array
        The array to reorder.

    Returns
    -------
    a : Numpy array
        The reordered array.

    """
    N, M = a.shape

    # Reorder the rows
    idx = np.argsort(-np.max(a, axis=1))
    a = a[idx, :]
    
    # Reorder the columns
    for i in range(N):
        
        if i == 0:
            a[i,:] = -np.sort(-a[i,:])
            
        elif i < M//2:
            sorted_row = -np.sort(-a[i,:])
            left = sorted_row[1:2*i:2]
            # Opposite side for coherent ordering around the diagonal value
            left = left[::-1]
            right = sorted_row[0:2*i:2]
            a[i, 0:i] = left
            a[i, i:2*i] = right
            a[i, 2*i:] = sorted_row[2*i:]
        
        elif i == M//2:
            sorted_row = np.sort(a[i,:])
            if M % 2 == 1:
                a[i, 0:M//2+1] = sorted_row[::2]
                a[i, M//2+1:] = sorted_row[-2::-2]
            else:
                a[i, 0:M//2] = sorted_row[::2]
                a[i, M//2:] = sorted_row[::-2]
            
        elif i < M:
            sorted_row = -np.sort(-a[i,:])
            left = sorted_row[1:2*(M-i):2]
            left = left[::-1]
            right = sorted_row[0:2*(M-i):2]
            a[i, 0:i-(M-i)] = sorted_row[2*(M-i):][::-1]
            a[i, i-(M-i):i] = left
            a[i, i:] = right
            
        else:
            a[i, :] = np.sort(a[i,:])
        
    return a

    
def intersection_over_union(subfolder, other_subfolder=None, save=False, filename=None):
    """
    Creates a plot showing the intersection over union for each clusters.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...
    other_subfolder : str, optional
        Another subfolder to compare assignment. If not provided, the comparison
        will be eprformed with the groundtruth. The default is None.
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

    tools._is_subfolder(subfolder)
    if other_subfolder is not None:
        tools._is_subfolder(other_subfolder)
        if other_subfolder[-1] != '/':
            other_subfolder += '/'
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
        
    if subfolder[-1] != '/':
        subfolder += '/'
    
    assignment = np.load(subfolder + 'assignment.npy')
    if other_subfolder is None:
        other_assignment = tools.get_groundtruth_attribute(subfolder, 'assignment')
    else:
        other_assignment = np.load(other_subfolder + '/assignment.npy')
        
    intersection, indices1, indices2 = tools.cluster_intersection_over_union(assignment, other_assignment)
    
    
    intersection = _reorder(intersection)

    fig = plt.figure(figsize=(12,12))
    ax = plt.gca()
    # sns.heatmap(intersection, annot=False, fmt='.2%', cmap='Greys', vmin=0., vmax=1.)
    matrix = ax.imshow(intersection, cmap='afmhot_r', vmin=0., vmax=1.)
    cax = fig.add_axes([ax.get_position().x1+0.02, ax.get_position().y0 , 0.04,
                        ax.get_position().height])
    plt.colorbar(matrix, cax=cax)
    if save:
        fig.savefig(subfolder + filename, bbox_inches='tight')
    plt.show()
    
    
def intersection_plot(subfolder, save=False, filename=None):
    
    tools._is_subfolder(subfolder)
    
    if save and filename is None:
        raise ValueError('Filename cannot be None if save is True.')
        
    if subfolder[-1] == '/':
        subfolder = subfolder.rsplit('/', 1)[0]
        
    directory = subfolder.rsplit('/', 1)[0]
    algorithm, _, _ = tools.extract_params_from_folder_name(directory)
    ref_assignment = np.load(subfolder + 'assignment.npy')
    
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
    
    max_intersections = []
    N_clusters = []
    
    for folder in subfolders:
        if folder == subfolder:
            continue
        assignment = np.load(folder + '/assignment.npy')
        intersection, _, _ = tools.cluster_intersections(ref_assignment, assignment,
                                                         algorithm)
        max_intersection = np.max(intersection, axis=1)
        max_intersections.append(max_intersection)
        N_clusters.append(len(intersection[0,:]))
        
    groundtruth_assignment = tools.get_groundtruth_attribute(directory, 'assignment')
    intersection, _, _ = tools.cluster_intersections(ref_assignment, groundtruth_assignment,
                                                     algorithm)
    max_intersection = np.max(intersection, axis=1)
    max_intersections.append(max_intersection)
    N_clusters.append(len(intersection[0,:]))
    
    N = len(max_intersections)
    
    fig, axes = plt.subplots(N, 1, figsize=(15, 15), sharex=True, sharey=True)
    
    for intersection, N_clust, ax in zip(max_intersections[0:-1], N_clusters[0:-1],
                                         axes[0:-1]):
        
        ax.hist(intersection, bins=100, range=(0, 1))
        ax.set(ylabel=f'{N_clust} clusters')
        ax.set(yscale='log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:.2%} %'))
        
    axes[-1].hist(max_intersections[-1], bins=100, color=['tab:red'], range=(0, 1))
    axes[-1].set(ylabel='Original')
    axes[-1].set(xlabel='Maximum clusters intersection')
    axes[-1].set(yscale='log')
    axes[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:.2%} %'))
    
    if save:
        fig.savefig(subfolder + '/' + filename, bbox_inches='tight')
    plt.show()


def completeness_homogeneity_plot(directory, save=False, filename=None):
    
    if directory[-1] != '/':
        directory += '/'
    
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = sorted(subfolders, reverse=True,
                        key=lambda x: int(x.rsplit('/', 1)[1].split('-', 1)[0]))
    
    homogeneities = []
    completenesses = []
    
    for subfolder in subfolders:
        homogeneity, completeness = tools.get_metrics(subfolder)
        homogeneities.append(homogeneity)
        completenesses.append(completeness)
        
    plt.figure()
    plt.plot(homogeneities, completenesses)
    plt.xlabel('Homogeneity score')
    plt.ylabel('Completeness score')
    plt.grid()
    if save:
        plt.savefig(directory + filename, bbox_inches='tight')
    plt.show()
    
    return homogeneities, completenesses

#%%

if __name__ == '__main__':

    # directory = 'Clustering_results'
    # for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
        # if 'DBSCAN' in folder:
            # for subfolder in [f.path for f in os.scandir(folder) if f.is_dir()]:
                # intersection_plot(subfolder, save=True, filename='intersection.pdf')
                # test = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples/220-clusters_4.125-eps'
                # intersection_plot(test, save=True, filename='test.pdf')

    # subfolder = 'Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/253-clusters_4.375-eps'
    # intersection_over_union(subfolder, save=True, filename='test1.pdf')
    # assignment = np.load(subfolder + '/assignment.npy')
    # if 'DBSCAN' in subfolder:
        # assignment = assignment[assignment != -1]

    directory = 'Clustering_results/full_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples'
    c, d = completeness_homogeneity_plot(directory, True, 'test')


    a, b = tools.get_metrics(directory + '/151-clusters_4.500-eps')
    print(f'Homogeneity : {a:.3f}')
    print(f'Completeness : {b:.3f}')

