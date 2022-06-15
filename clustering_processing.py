#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:53:54 2022

@author: cyrilvallez
"""
from tqdm import tqdm
import os

from helpers import utils
from clustering import tools

# directory = 'Clustering_results'
# for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
    # counts = cluster_size_violin(folder, save=True, filename='sizes_violin.pdf')
    # cluster_size_evolution(folder, save=False, filename='sizes.pdf')
    # cluster_diameter_violin(folder, save=True, filename='sizes_violin.pdf')
        
directory = 'Clustering_results'
for folder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
    tools.save_diameters(folder)
    
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