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
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import homogeneity_score, completeness_score
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
    
    
def compute_assignment_groundtruth(algorithm, metric, partition='full_dataset'):

    """
    Find and save the assignment of memes inside each "real" clusters (from the
    groundtruths we have).

    Parameters
    ----------
    algorithm : str
        Algorithm name used for extracting the features from images.
    metric : str
        Metric to use for later distance computation (e.g diameters).
    partition : str
        The desired partition. Either `full_dataset` or `clean_dataset`.

    Returns
    -------
    assignment : Numpy array
        The clustering assignment of the groundtruth.
    
    """
    assert partition in ALLOWED_PARTITIONS
    algorithm = '_'.join(algorithm.split())
    folder = f'Clustering_results/{partition}/{metric}_GT_{algorithm}'
    
    _, mapping = extract_features_from_folder_name(folder)
    
    # Find the count of memes inside each "real" clusters (from the groundtruths)
    identifiers = []
    for name in mapping:
        identifier = name.rsplit('/', 1)[1].split('_', 1)[0]
        if '.' in identifier:
            identifier = name.rsplit('/', 1)[1].rsplit('.', 1)[0]
        identifiers.append(identifier)
        
    encoder = LabelEncoder()
    assignment = encoder.fit_transform(identifiers)
    
    os.makedirs(folder, exist_ok=True)
    np.save(folder + '/assignment.npy', assignment)
    
    return assignment
 
        
def _compute_cluster_diameters(subfolder, quantile=1.):
    """
    Compute the diameter of each clusters.

    Parameters
    ----------
    subfolder : str
        Subfolder of a clustering experiment, containing the assignments,
        representatives images etc...
    quantile : float, optional
        The quantile on which to base the diameters. Give 1 for the maximum
        of the distances. The default is 1.

    Returns
    -------
    Numpy array
        The diameter of each cluster.

    """
    
    _is_subfolder(subfolder)
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    _, _, distances = extract_features_from_folder_name(subfolder, return_distances=True)
        
    assignments = np.load(subfolder + 'assignment.npy')
        
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


def _compute_cluster_centroids(subfolder):
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
    
    _, metric, _ = extract_params_from_folder_name(subfolder)
    features, _ = extract_features_from_folder_name(subfolder)
    assignments = np.load(subfolder + 'assignment.npy')
    
    unique, counts = np.unique(assignments, return_counts=True)

    engine = NearestCentroid(metric=metric)
    engine.fit(features, assignments)
    
    # They are already sorted correctly with respect to the cluster indices
    return engine.centroids_



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
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    try:
        diameters = np.load(subfolder + f'diameters_{quantile:.2f}.npy')
    except FileNotFoundError:
        diameters = _compute_cluster_diameters(subfolder, quantile=quantile)
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
    
    if subfolder[-1] != '/':
        subfolder += '/'
        
    try:
        centroids = np.load(subfolder + 'centroids.npy')
    except FileNotFoundError:
        centroids = _compute_cluster_centroids(subfolder)
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
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        
        diameters = _compute_cluster_diameters(subfolder, quantile=quantile)
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
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        
        centroids = _compute_cluster_centroids(subfolder)
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
    algorithm = '_'.join(algorithm.split())
    if attribute == 'assignment':
        try:
            groundtruth = np.load(f'Clustering_results/{partition}/{metric}_GT_{algorithm}/{attribute}.npy') 
        except FileNotFoundError:
            groundtruth = compute_assignment_groundtruth(algorithm, metric,
                                                         partition=partition)
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


def clean_dataset(mapping):
    
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
        

def get_metrics(subfolder):
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
    homogeneity : float
        The homogeneity score. It measures if each cluster contains only
        members of a single class.
    completeness : float
        The completeness score. It measures if all members of a given class
        are assigned to the same cluster.

    """
    
    _is_subfolder(subfolder)

    if subfolder[-1] != '/':
        subfolder += '/'
        
    assignments = np.load(subfolder + 'assignment.npy')
    groundtruth_assignment = get_groundtruth_attribute(subfolder, 'assignment')
    
    homogeneity = homogeneity_score(groundtruth_assignment, assignments)
    completeness = completeness_score(groundtruth_assignment, assignments)
    
    return homogeneity, completeness
#%%

if __name__ == '__main__':
    
    
    
    # algorithm = 'Dhash 64 bits'
    # metric = 'hamming'
    # compute_assignment_groundtruth(algorithm, metric)
    # get_cluster_diameters('Clustering_results/hamming_GT_Dhash_64_bits', True)
    # directory = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples'
    # foo = cluster_diameter_violin(directory, save=True, filename='test34.pdf')
    
    # directory = 'Clustering_results'
    # for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
        # if 'DBSCAN' in folder:
            # for subfolder in [f.path for f in os.scandir(folder) if f.is_dir()]:
                # save_extremes(subfolder)
    
    def find_templates(assignment, mapping):
        
        identifiers = []
        for name in mapping:
            identifier = name.rsplit('/', 1)[1].split('_', 1)[0]
            if '.' in identifier:
                identifier = name.rsplit('/', 1)[1].rsplit('.', 1)[0]
            identifiers.append(identifier)
                
        identifiers = np.array(identifiers)
                
        cluster_names = []
        for idx in np.unique(assignment):
            cluster = identifiers[assignment == idx]
            assert len(np.unique(cluster)) == 1
            cluster_names.append(cluster[0])
                
        return np.array(cluster_names)
    
    """
    path = 'Clustering_results/euclidean_GT_SimCLR_v2_ResNet50_2x'
    assignment = np.load(path + '/assignment.npy')
    diameters = get_cluster_diameters(path, quantile=0.5)
    _, mapping = extract_features_from_folder_name(path)
    templates = find_templates(assignment, mapping)
    sorting = np.argsort(diameters)
    bad = templates[sorting[-10:]]
    """
    
    """
    path = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples/233-clusters_4.000-eps'
    assignment = np.load(path + '/assignment.npy')
    _, mapping = extract_features_from_folder_name(path)
    
    identifiers = []
    for name in mapping:
        identifier = name.rsplit('/', 1)[1].split('_', 1)[0]
        if '.' in identifier:
            identifier = name.rsplit('/', 1)[1].rsplit('.', 1)[0]
        identifiers.append(identifier)
            
    identifiers = np.array(identifiers)
    
    
    # meme = 'who-would-win'
    meme = 'skyrim-100'
    cond = identifiers == meme
    foo = assignment[cond]
    bar = (assignment == -1).sum()
    """
    
    directory1 = 'Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/253-clusters_4.375-eps'
    gt_assignment1 = get_groundtruth_attribute(directory1, 'assignment')
    assignment1 = np.load(directory1 + '/assignment.npy')
    
    directory2 = 'Clustering_results/full_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/268-clusters_4.375-eps'
    gt_assignment2 = get_groundtruth_attribute(directory2, 'assignment')
    assignment2 = np.load(directory2 + '/assignment.npy')
    
    features, mapping = extract_features_from_folder_name(directory2)
    
    
    
    #%%
    """
    import time
    from scipy.spatial.distance import squareform
    from sklearn.manifold import TSNE

    folder = 'Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples'
    features, mapping, distances = extract_features_from_folder_name(folder, return_distances=True)
    # Reshape the distances as a symmetric matrix
    distances = squareform(distances)

    t0 = time.time()
    embedding = TSNE(metric='precomputed', learning_rate='auto', method='exact')
    foo = embedding.fit_transform(distances)
    dt = time.time() - t0
    """



    