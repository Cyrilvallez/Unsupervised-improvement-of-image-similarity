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
from sklearn.metrics import homogeneity_completeness_v_measure

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
    
    # Giving other metric than euclidean or L1, this just compute the same as
    # euclidean (mean), thus we do not bother giving the option since this
    # may give unexpected results
    engine = NearestCentroid(metric='euclidean')
    engine.fit(features, assignments)
    
    # They are already sorted correctly with respect to the cluster indices
    return engine.centroids_


def cluster_intersection_over_union(assignment1, assignment2):
    """
    Compute the intersection over union matrix of 2 cluster assignments. Each 
    entry [i,j] represents the intersection over union of cluster i in `assignment1`
    with cluster j in `assignment2`.

    Parameters
    ----------
    assignment1 : Numpy array
        The first cluster assignments.
    assignment2 : Numpy array
        TThe second cluster assignments.

    Returns
    -------
    intersection : Numpy array
        The intersection matrix.
    indices1 : Numpy array
        Indices of clusters corresponding to the columns.
    indices2 : Numpy array
        Indices of clusters corresponding to the rows.

    """
    
    indices1 = np.unique(assignment1)
    indices2 = np.unique(assignment2)
    
    # Do not take cluster of outliers into account, if it exists
    indices1 = indices1[indices1 != -1]
    indices2 = indices2[indices2 != -1]
    
    intersection = np.empty((len(indices1), len(indices2)))
    
    for i, index1 in enumerate(indices1):
        
        cluster1 = np.argwhere(assignment1 == index1).flatten()

        for j, index2 in enumerate(indices2):
            
            cluster2 = np.argwhere(assignment2 == index2).flatten()
            inter = np.intersect1d(cluster1, cluster2, assume_unique=True)
            union = np.union1d(cluster1, cluster2)
            intersection[i,j] = len(inter)/len(union)

    return intersection, indices1, indices2


def scores(groundtruth_assignment, assignment):
    """
    Compute the homogeneity and completeness of the clustering experiment in
    `subfolder` against the groundtruths.

    Parameters
    ----------
    groundtruth_assignment : Numpy array
        The groundtruth cluster assignments.
    assignment : Numpy array
        The experimental cluster assignments.

    Returns
    -------
    h : float
        The homogeneity score. It measures if each cluster contains only
        members of a single class.
    c : float
        The completeness score. It measures if all members of a given class
        are assigned to the same cluster.
    v : float
        The v-measure score. This is the harmonic mean of homogeneity and
        completeness.
        
    """
    
    h, c, v = homogeneity_completeness_v_measure(groundtruth_assignment, assignment)
    
    return h, c, v



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

