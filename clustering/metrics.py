#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:30:42 2022

@author: cyrilvallez
"""

import torch
import torch.nn.functional as F
import scipy.spatial.distance as D
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# All metrics inside this script are computed using euclidean L2 distance 
# between points. Thus if you are clustering your features (images) using
# a different distance, those metrics may not be coherent.
# =============================================================================

def cluster_diameters(features, assignments, quantile=1.): 
    """
    Compute the diameters of each cluster. The diameters are based on the
    given quantile of the pairwise euclidean distances between points
    inside the cluster. Giving 1 for the quantile returns the maximum
    of the pairwise distances.

    Parameters
    ----------
    features : Numpy array
        The features corresponding to the images being clustered.
    assignments : Numpy array
        The cluster assignment for each feature (image).
    quantile : float, optional
        The quantile on which to base the diameters. Give 1 for the maximum
        of the distances. The default is 1.

    Returns
    -------
    Numpy array
        The diameter of each cluster (sorted in increasing order of cluster 
        indices).

    """
        
    diameters = []
    
    for cluster_idx in np.unique(assignments):
        indices = assignments == cluster_idx
        
        # If the cluster has a lot of representants, try computing on gpu
        if indices.sum() > 4000 and torch.cuda.is_available():
            cluster_features = torch.tensor(features[indices], device='cuda')
            distances = F.pdist(cluster_features).cpu().numpy()
        else:
            distances = D.pdist(features[indices])
        
        # If the cluster contains only 1 image
        if len(distances) == 0:
            diameters.append(0.)
        else:
            diameters.append(np.quantile(distances, quantile))
        
    return np.array(diameters)


def cluster_centroids(features, assignments):
    """
    Compute the centroids of each cluster. The centroids for the samples
    corresponding to each cluster is the point from which the sum of the distances
    (according to the metric) of all samples that belong to that particular cluster
    are minimized. We define it as the mean of the features corresponding to
    the current cluster. Thus this function is coherent only if using euclidean
    distance for measuring distances between features (otherwise there are no
    guarantees that the mean minimizes the distance used).

    Parameters
    ----------
    features : Numpy array
        The features corresponding to the images being clustered.
    assignments : Numpy array
        The cluster assignment for each feature (image).

    Returns
    -------
    Numpy array
        The centroids of each cluster (sorted in increasing order of cluster 
        indices).

    """

    engine = NearestCentroid(metric='euclidean')
    engine.fit(features, assignments)
    
    # They are already sorted correctly with respect to the cluster indices
    return engine.centroids_


def assignment_groundtruth(mapping):
    
    # Find the count of memes inside each "real" clusters (from the groundtruths)
    identifiers = []
    for name in mapping:
        identifier = name.rsplit('/', 1)[1].split('_', 1)[0]
        if '.' in identifier:
            identifier = name.rsplit('/', 1)[1].rsplit('.', 1)[0]
        identifiers.append(identifier)
        
    encoder = LabelEncoder()
    assignment = encoder.fit_transform(identifiers)
    
    return assignment


def outside_cluster_separation(features, assignments, centroids=None):
    """
    Compute the minimum of the distances between each cluster centroid and points
    lying outside this cluster. In a sense, this measures how well
    the separation between all clusters is. Note that this uses euclidean
    distance.

    Parameters
    ----------
    features : Numpy array
        The features corresponding to the images being clustered.
    assignments : Numpy array
        The cluster assignment for each feature (image).
    centroids : Numpy array, optional
        The centroids of the clusters. If not given, this will compute them.
        The default is None.

    Returns
    -------
    Numpy array
        The minimal distances (sorted in increasing order of cluster indices).

    """
    
    if centroids is None:
        centroids = cluster_centroids(features, assignments)
    
    distances = []
    for i, cluster in enumerate(np.unique(assignments)):
        indices = assignments != cluster
        distance = np.min(np.linalg.norm(features[indices] - centroids[i], axis=1))
        distances.append(distance)
        
    return np.array(distances)


def inside_cluster_dispersion(features, assignments, centroids=None):
    """
    Compute the mean of the distances between all points inside a cluster
    and the cluster centroid. In a sense, this measures spatial dispersion
    inside each cluster. Note that this uses euclidean distance.

    Parameters
    ----------
    features : Numpy array
        The features corresponding to the images being clustered.
    assignments : Numpy array
        The cluster assignment for each feature (image).
    centroids : Numpy array, optional
        The centroids of the clusters. If not given, this will compute them.
        The default is None.

    Returns
    -------
    Numpy array
        The mean distances (sorted in increasing order of cluster indices).

    """
    
    if centroids is None:
        centroids = cluster_centroids(features, assignments)
    
    distances = []
    for i, cluster in enumerate(np.unique(assignments)):
        indices = assignments == cluster
        distance = np.mean(np.linalg.norm(features[indices] - centroids[i], axis=1))
        distances.append(distance)
        
    return np.array(distances)



