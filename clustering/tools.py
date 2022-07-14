#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:24:02 2022

@author: cyrilvallez
"""

from PIL import Image
import numpy as np
import os
from scipy.spatial.distance import pdist
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from helpers import utils
from clustering import metrics

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
      

def save_representatives(subfolder):
    """
    Save representatives of each clusters from the cluster assignments.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...

    Returns
    -------
    None.

    """
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    os.makedirs(subfolder + 'representatives', exist_ok=True)
    
    _, mapping = extract_features_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
    
    np.random.seed(112)
    
    for cluster_idx in np.unique(assignments):
    
        images = mapping[assignments == cluster_idx]
        if len(images) > 10:
            representatives = np.random.choice(images, size=10, replace=False)
        else:
            representatives = images
        
        representation = cluster_representation(representatives)
        representation.save(subfolder + f'representatives/{cluster_idx}.png')
        
        
def save_extremes(subfolder):
    """
    Concatenate and save images from the same cluster, by pulling 5 closest
    and 5 furthest from the cluster centroid.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...

    Returns
    -------
    None.

    """
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    os.makedirs(subfolder + 'extremes', exist_ok=True)
    
    assignment = np.load(subfolder + 'assignment.npy')
    centroids = get_cluster_centroids(subfolder)
    
    features, mapping = extract_features_from_folder_name(subfolder)
    
    extremes = extremes_from_centroids(features, mapping, assignment, centroids)
    cluster_indices = np.unique(assignment)
    
    for index, extreme in zip(cluster_indices, extremes):
        representation = cluster_representation(extreme)
        representation.save(subfolder + f'extremes/{index}.png')
        
        
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
        
    if os.path.dirname(os.path.dirname(directory)) != 'Clustering_results':
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
    
    # If this is a groundtruth folder, we skip the checks because the file structure 
    # is different
    if 'GT' in subfolder:
        return 
    
    if subfolder[-1] == '/':
        subfolder = subfolder.rsplit('/', 1)[0]
        
    split = subfolder.rsplit('/', 3)
    if split[0] != 'Clustering_results' or len(split) != 4:
        raise ValueError('The subfolder you provided is not valid.')
        

ALLOWED_PARTITIONS = [
    'full_dataset',
    'clean_dataset',
    ]
        
def get_dataset(partition, algorithm):
    """
    Extract the features and mapping corresponding to the wanted `partition` of
    the kaggle memes dataset.

    Parameters
    ----------
    partition : str
        The desired partition. Either `full_dataset` or `clean_dataset`.
    algorithm : str
        Algorithm used to extract the features.

    Returns
    -------
    features : Numpy array
        The features corresponding to the images.
    mapping : Numpy array
        Mapping from feature index to actual image (as a path name).

    """
    assert partition in ALLOWED_PARTITIONS
    
    dataset1 = 'Kaggle_memes'
    dataset2 = 'Kaggle_templates'
    
    # Load features and mapping to actual images
    features, mapping = utils.combine_features(algorithm, dataset1, dataset2,
                                           to_bytes=False)
    if partition == 'clean_dataset':
        indices = clean_dataset(mapping)
        features, mapping = features[indices], mapping[indices]
        
    return features, mapping

        
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
        
    metric = directory.rsplit('/', 1)[1].split('_', 1)[0]
    partition = directory.split('/')[1]
    
    return algorithm, metric, partition


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
    
    algorithm, metric, partition = extract_params_from_folder_name(directory)
    
    # Load features and mapping to actual images
    features, mapping = get_dataset(partition, algorithm)
    
    if return_distances:
        identifier = '_'.join(algorithm.split(' '))
        try:  
            distances = np.load(f'Clustering_results/{partition}/distances_{identifier}_{metric}.npy')
        except FileNotFoundError:
            distances = pdist(features, metric=metric)
            np.save(f'Clustering_results/{partition}/distances_{identifier}_{metric}.npy', distances)
        
        return features, mapping, distances
    
    else:
        return features, mapping   
    
    
def compute_assignment_groundtruth(mapping):
    """
    Find and save the assignment of memes inside each "real" clusters (from the
    groundtruths we have).

    Parameters
    ----------
    mapping : Numpy array
        Mapping from feature index to actual image (as a path name).

    Returns
    -------
    assignment : Numpy array
        The clustering assignment of the groundtruth.
    
    """
    
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


def get_cluster_diameters(subfolder, quantile=1., save_if_absent=True):
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
    diameters : Numpy array
        The diameters.

    """
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    features, _ = extract_features_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
        
    try:
        diameters = np.load(subfolder + f'diameters_{quantile:.2f}.npy')
    except FileNotFoundError:
        diameters = metrics.cluster_diameters(features, assignments, quantile)
        if save_if_absent:
            np.save(subfolder + f'diameters_{quantile:.2f}.npy', diameters)
            
    return diameters


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
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    features, _ = extract_features_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
        
    try:
        centroids = np.load(subfolder + 'centroids.npy')
    except FileNotFoundError:
        centroids = metrics.cluster_centroids(features, assignments)
        if save_if_absent:
            np.save(subfolder + 'centroids.npy', centroids)
            
    return centroids
        

def save_diameters(directory, quantile=1.):
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
    
    _is_directory(directory)
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        
        features, _ = extract_features_from_folder_name(subfolder)
        assignments = np.load(subfolder + '/assignment.npy')
        diameters = metrics.cluster_diameters(features, assignments, quantile)
        np.save(subfolder + f'/diameters_{quantile:.2f}.npy', diameters)
        

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
    
    _is_directory(directory)
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        
        features, _ = extract_features_from_folder_name(subfolder)
        assignments = np.load(subfolder + '/assignment.npy')
        centroids = metrics.cluster_centroids(features, assignments)
        np.save(subfolder + '/centroids.npy', centroids)
    
    
def get_groundtruth_attribute(directory, attribute, quantile=1.):
    """
    Get the groundtruth `attribute` corresponding to the actual `directory`.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.
    attribute : str
        Either `assignment`, `diameters` or `centroids`.
    quantile : float, optional
        Only used if `attribute` is `diameters`.The quantile on which to base
        the diameters. Give 1 for the maximum of the distances. The default is 1.

    Raises
    ------
    ValueError
        If `attribute` is not correct.

    Returns
    -------
    groundtruth : Numpy array
        The array corresponding to the groundtruth of the attribute.

    """
    
    algorithm, metric, partition = extract_params_from_folder_name(directory)
    _, mapping = extract_features_from_folder_name(directory)
    algorithm = '_'.join(algorithm.split())
    if attribute == 'assignment':
        try:
            groundtruth = np.load(f'Clustering_results/{partition}/{metric}_GT_{algorithm}/{attribute}.npy') 
        except FileNotFoundError:
            groundtruth = compute_assignment_groundtruth(mapping)
            folder = f'Clustering_results/{partition}/{metric}_GT_{algorithm}'
            os.makedirs(folder, exist_ok=True)
            np.save(folder + '/assignment.npy', groundtruth)
            
    elif attribute == 'diameters':
        groundtruth = get_cluster_diameters(f'Clustering_results/{partition}/{metric}_GT_{algorithm}',
                                            quantile=quantile)
    elif attribute == 'centroids':
        groundtruth = get_cluster_centroids(f'Clustering_results/{partition}/{metric}_GT_{algorithm}')
    else:
        raise ValueError('The attribute is not correct.')
        
    return groundtruth
        
    
def extremes_from_centroids(features, mapping, assignment, centroids):
    """
    Compute the 5 extremes (closest and furthest) points from the centroid
    for each cluster. Note that the distance used is always euclidean.

    Parameters
    ----------
    features : Numpy array
        Features extracted from the images.
    mapping : Numpy array
        Mapping from indices to image path.
    assignment : Numpy array
        The cluster assignment.
    centroids : Numpy array
        The centroid of each cluster.

    Returns
    -------
    extremes : list of list
        The path to the most extremes images inside each cluster.

    """
    extremes = []
    
    for cluster_idx in np.unique(assignment):
        
        indices = assignment == cluster_idx
        cluster_features = features[indices]
        cluster_images = mapping[indices]
        centroid = centroids[cluster_idx]
        
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        sorting = np.argsort(distances)
        cluster_images = cluster_images[sorting]
        if len(cluster_images) <= 10:
            extremes.append(cluster_images.tolist())
        else:
            extreme = cluster_images[0:5].tolist() + cluster_images[-5:].tolist()
            extremes.append(extreme)
        
    return extremes


def clean_dataset(mapping):
    """
    Returns the indices corresponding to only the perceptualy identical memes
    in the Kaggle dataset.

    Parameters
    ----------
    mapping : Numpy array
        Mapping from indices to image path.

    Returns
    -------
    indices : list
        The indices.

    """
    
    templates_to_remove = [
        'zuckerberg',
        'harold',
        'netflix-adaptation',
        'shrek',
        'so-glad',
        'who-would-win',
        'you-vs-the-guy',
        'skyrim-100',
        ]
    
    indices = []
    for i, name in enumerate(mapping):
        identifier = name.rsplit('/', 1)[1].split('_', 1)[0]
        if '.' in identifier:
            identifier = name.rsplit('/', 1)[1].rsplit('.', 1)[0]
        
        if identifier not in templates_to_remove:
            indices.append(i)
            
    return indices


def get_scores(subfolder):
    """
    Compute the homogeneity and completeness of the clustering experiment in
    `subfolder` against the groundtruths.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...

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
    
    _is_subfolder(subfolder)

    if subfolder[-1] != '/':
        subfolder += '/'
        
    assignments = np.load(subfolder + 'assignment.npy')
    groundtruth_assignment = get_groundtruth_attribute(subfolder, 'assignment')
    
    return metrics.scores(groundtruth_assignment, assignments)
