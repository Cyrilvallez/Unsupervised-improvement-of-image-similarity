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
    centroids = np.load(subfolder + 'centroids.npy')
    
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
    
    
def compute_assignment_groundtruth(algorithm, metric):

    """
    Find and save the assignment of memes inside each "real" clusters (from the
    groundtruths we have).

    Parameters
    ----------
    algorithm : str
        Algorithm name used for extracting the features from images.
    metric : str
        Metric to use for later distance computation (e.g diameters).

    Returns
    -------
    None.
    
    """
    
    algorithm = '_'.join(algorithm.split())
    folder = f'Clustering_results/{metric}_GT_{algorithm}'
    
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
    
    # If this is a groundtruth, we skip the check because the file structure is
    # different
    if not ('GT' in subfolder):
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
    
    # If this is a groundtruth, we skip the check because the file structure is
    # different
    if not ('GT' in subfolder):
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
    
    assert ('/' not in identifier), 'Identifier cannot be a path'
    
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
    
    assert ('/' not in identifier), 'Identifier cannot be a path'
    
    for subfolder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
        
        attribute = func(subfolder)
        np.save(subfolder + '/' + identifier + '.npy', attribute)
        

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
    
    
def get_groundtruth_attribute(directory, attribute):
    """
    Get the groundtruth `attribute` corresponding to the actual `directory`.

    Parameters
    ----------
    directory : str
        The directory where the experiment is contained.
    attribute : str
        Either `assignment`, `diameters` or `centroids`.

    Raises
    ------
    ValueError
        If `attribute` is not correct.

    Returns
    -------
    groundtruth : Numpy array
        The array corresponding to the groundtruth of the attribute.

    """
    
    algorithm, metric = extract_params_from_folder_name(directory)
    algorithm = '_'.join(algorithm.split())
    if attribute == 'assignment':
        try:
            groundtruth = np.load(f'Clustering_results/{metric}_GT_{algorithm}/{attribute}.npy') 
        except:
            compute_assignment_groundtruth(algorithm, metric)
            groundtruth = np.load(f'Clustering_results/{metric}_GT_{algorithm}/{attribute}.npy')
    elif attribute == 'diameters':
        groundtruth = get_cluster_diameters(f'Clustering_results/{metric}_GT_{algorithm}/{attribute}.npy')
    elif attribute == 'centroids':
        groundtruth = get_cluster_centroids(f'Clustering_results/{metric}_GT_{algorithm}/{attribute}.npy')
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


def cluster_intersections(assignment1, assignment2, algorithm):
    """
    Compute the intersection matrix of 2 cluster assignments. Each row 
    represents the intersection percentage of cluster i in `assignment1` with
    all other clusters in `assignment2`. Thus all rows sum to 1.

    Parameters
    ----------
    assignment1 : Numpy array
        The first cluster assignments.
    assignment2 : Numpy array
        The second cluster assignments.
    algorithm : str
        The algorithm used to extract the features.

    Returns
    -------
    intersection_percentage : Numpy array
        The intersection matrix.
    cluster_indices1 : Numpy array
        Indices of clusters corresponding to the columns.
    cluster_indices2 : Numpy array
        Indices of clusters corresponding to the rows.

    """
    
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

        
#%%

if __name__ == '__main__':
    
    """
    algorithm = 'SimCLR v2 ResNet50 2x'
    metric = 'cosine'
    folder = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/268-clusters_4.375-eps'
    folder2 = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_5_samples/306-clusters_4.250-eps'
    assignment1 = np.load(folder + '/assignment.npy')
    assignment2 = np.load(folder2 + '/assignment.npy')
    intersection, indices1, indices2 = clusters_intersection(assignment1,
                                                             assignment2, algorithm)
    
    # intersection, indices1, indices2 = intersection_plot(folder, folder2, save=True, filename='test.pdf')

    
    indices2 = indices2[0:len(indices1)]
    intersection = intersection[0:len(indices1), 0:len(indices1)]
    plt.figure(figsize=(8,8))
    plt.hexbin(indices2, indices1, intersection)
    plt.savefig('test_hexbin.pdf')
    
    
    foo = intersection[intersection > 0]
    
    plt.figure()
    plt.hist(foo.flatten(), bins=50)
    # plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'${x:.2%}$ %'))
    
    print((intersection == 1).sum())
    """
    
    # algorithm = 'Dhash 64 bits'
    # metric = 'hamming'
    # compute_assignment_groundtruth(algorithm, metric)
    # get_cluster_diameters('Clustering_results/hamming_GT_Dhash_64_bits', True)
    # directory = 'Clustering_results/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples'
    # foo = cluster_diameter_violin(directory, save=True, filename='test34.pdf')
    
    directory = 'Clustering_results'
    for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
        if 'DBSCAN' in folder:
            for subfolder in [f.path for f in os.scandir(folder) if f.is_dir()]:
                save_extremes(subfolder)
            
    
        

    