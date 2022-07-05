#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:58:28 2022

@author: cyrilvallez
"""

import numpy as np
import torch
from torch.nn.functional import pdist
import itertools
import os
from sklearn.neighbors import NearestCentroid

import extractor
from clustering import tools
from finetuning.simclr import SimCLR


def compute_diameters(distances, quantile=1.):
        
    assignments = np.load(('Clustering_results/clean_dataset/'
                           'euclidean_GT_SimCLR_v2_ResNet50_2x/assignment.npy'))
        
    # Mapping from distance matrix indices to condensed representation index
    N = len(assignments)
    
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
            diameters.append(np.quantile(cluster_distances, quantile))
        
    return np.array(diameters)


def compute_centroids(features):
    
    assignments = np.load(('Clustering_results/clean_dataset/'
                           'euclidean_GT_SimCLR_v2_ResNet50_2x/assignment.npy'))
    
    unique, counts = np.unique(assignments, return_counts=True)

    engine = NearestCentroid(metric='euclidean')
    engine.fit(features, assignments)
    
    # They are already sorted correctly with respect to the cluster indices
    return engine.centroids_



if __name__ == '__main__':

    transforms = extractor.SIMCLR_TRANSFORMS
    dataset = extractor.all_memes_dataset(transforms)

    epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    
    paths = ['first_test_models/2022-06-30_20:03:28/epoch_{a}.pth' for a in epochs[1:]]
    
    all_diameters = []
    all_diameters.append(
        # Fake directory, just to get the groundtruth from the clean dataset and original
        # model
        tools.get_groundtruth_attribute(
            'Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples',
            'diameters', quantile=1.))
    
    all_centroids = []
    all_centroids.append(
        # Fake directory, just to get the groundtruth from the clean dataset and original
        # model
        tools.get_groundtruth_attribute(
            'Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples',
            'centroids'))
    
    for path in paths:
        
        model = SimCLR.load_encoder(path)
        features, mapping = extractor.extract_neural(model, dataset, batch_size=64)
        indices = tools.clean_dataset(mapping)
        features, mapping = features[indices], mapping[indices]
        features = torch.tensor(features).cuda()
        distances = pdist(features).cpu().numpy()
        features = features.cpu().numpy()

        diameters = compute_diameters(distances, quantile=1.)
        all_diameters.append(diameters)
        
        centroids = compute_centroids(features)
        all_centroids.append(centroids)
       
    os.makedirs('Finetuning_eval', exist_ok=True)

    for diameters, epoch in zip(all_diameters, epochs):
        np.save(f'Finetuning_eval/diameters_epochs_{epoch}.npy', diameters)
        
    for centroids, epoch in zip(all_diameters, epochs):
        np.save(f'Finetuning_eval/centroids_epochs_{epoch}.npy', diameters)
 


