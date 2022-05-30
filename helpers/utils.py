#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:02:28 2022

@author: cyrilvallez
"""
import numpy as np
import json
import argparse
import os
from PIL import Image

import extractor.datasets as datasets
from extractor.neural import MODEL_LOADER
from extractor.perceptual import NAME_TO_ALGO


def load_features(method_name, dataset_name):
    """
    Load features from a given method identifier and dataset.

    Parameters
    ----------
    method_name : str
        The identifier of the method from which the features were extracted.
    dataset_name : str
        Identifier of the dataset from which the features were extracted.

    Raises
    ------
    ValueError
        If any identifier is erroneous.

    Returns
    -------
    features : Numpy array
        The features.
    mapping : Numpy array
        A mapping from indices to image names.

    """
    
    if (method_name not in MODEL_LOADER.keys() and not \
        any(method_name in a for a in NAME_TO_ALGO.keys())):
        raise ValueError(f'Method name must be one of {*MODEL_LOADER.keys(),} or {*NAME_TO_ALGO.keys(),}.')
    
    if dataset_name not in datasets.VALID_DATASET_NAMES:
        raise ValueError(f'The dataset name must be one of {*datasets.VALID_DATASET_NAMES,}.')
        
    path = 'Features/' + dataset_name + '-' + '_'.join(method_name.split(' '))

    features = np.load(path + '_features.npy')
    mapping = np.load(path + '_map_to_names.npy', allow_pickle=True)
    
    return features, mapping


def combine_features(method_name, dataset_name, other_dataset_name='Flickr500K'):
    """
    Combine two set of features into a single dataset. For example it combines
    any default dataset with the 500K distractor images from Flickr dataset.

    Parameters
    ----------
    method_name : str
        The identifier of the method from which the features were extracted.
    dataset_name : str
        Identifier of the first (smallest) dataset to merge.
    other_dataset_name : str, optional
        Identifier of the second (biggest) dataset to merge. The default is 'Flickr500K'.

    Returns
    -------
    features : Numpy array
        The features.
    mapping : Numpy array
        A mapping from indices to image names.

    """
    
    N1 = datasets.DATASET_DIMS[dataset_name]
    N2 = datasets.DATASET_DIMS[other_dataset_name]
    
    # Load the (supposedly) smaller dataset and use it to get dimension of features
    features1, mapping1 = load_features(method_name, dataset_name)
    M = features1.shape[1]
    
    # Initialize the big array to ensure to load directly data into it to avoid
    # very memory consuming copies
    features = np.empty((N1 + N2, M), dtype=features1.dtype)
    mapping = np.empty((N1 + N2), dtype=object)
    
    features[0:N1, :] = features1
    mapping[0:N1] = mapping1
    
    # In case it takes a lot of memory already at this point
    del features1, mapping1
    
    features[N1:, :], mapping[N1:] = load_features(method_name, other_dataset_name)
    
    return features, mapping


def recall(neighbors, mapping_db, mapping_query):
    """
    Compute the recall from the results of the search.

    Parameters
    ----------
    neighbors : Numpy array
        Array containing indices of closest match for each query image.
    mapping_db : Numpy array
        The mapping from indices to image name in the database.
    mapping_query : Numpy array
        The mapping from indices to image name in the queries.

    Returns
    -------
    recall : Float
        The recall value for the search.
    correct : Numpy array
        Array of correct/incorrect search for each query.

    """
    
    # Extract portions of names that should be similar in image names
    search_identifiers = np.array([name.rsplit('/', 1)[1].split('_', 1)[0] for name in mapping_query])
    db_identifiers = np.array([name.rsplit('/', 1)[1].rsplit('.', 1)[0] for name in mapping_db])
    
    shape = neighbors.shape
    names = db_identifiers[neighbors.flatten()].reshape(shape)
    
    query_identifiers = np.expand_dims(search_identifiers, axis=1)
    
    correct = (names == query_identifiers).sum(axis=1)
    recall = correct.sum()/len(correct)
    
    return recall, correct


def normalize(x, order=2):
    """
    Normalize each row of a matrix.

    Parameters
    ----------
    x : Numpy array
        The matrix to normalize.
    order : float, optional
        Order for normalization. The default is 2.

    Returns
    -------
    The normalized vector.

    """
    
    norm = np.linalg.norm(x, axis=1, ord=order)
    # avoid division by 0
    norm[norm == 0] = 1
    
    return x/np.expand_dims(norm, axis=1)


def save_dictionary(dictionary, filename):
    """
    Save a dictionary to disk as json file.

    Parameters
    ----------
    dictionary : Dictionary
        The dictionary to save.
    filename : str
        Filename to save the file.

    Returns
    -------
    None.

    """
    
    # Make sure the path exists, and creates it if this is not the case
    dirname = os.path.dirname(filename)
    exist = os.path.exists(dirname)
    
    if not exist:
        os.makedirs(dirname)
    
    with open(filename, 'w') as fp:
        json.dump(dictionary, fp, indent='\t')
        
        
def load_dictionary(filename):
    """
    Load a json file and return a dictionary.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    data : Dictionary
        The dictionary representing the file.

    """
    
    file = open(filename)
    data = json.load(file)
    return data


def parse_input():
    """
    Create a parser for command line arguments in order to get the experiment
    folder. Also check that this folder is valid, in the sense that it is not 
    already being used.
    
    Raises
    ------
    ValueError
        If the experiment name is already taken or not valid.

    Returns
    -------
    save_folder : str
        The path to the folder for saving the experiment.

    """
    # Force the use of a user input at run-time to specify the path 
    # so that we do not mistakenly reuse the path from previous experiments
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('experiment_folder', type=str, help='A name for the experiment')
    args = parser.parse_args()
    experiment_folder = args.experiment_folder

    if '/' in experiment_folder:
        raise ValueError('The experiment name must not be a path. Please provide a name without any \'/\'.')

    results_folder = 'Results/'
    save_folder = results_folder + experiment_folder 

    # Check that it does not already exist and contain results
    if experiment_folder in os.listdir(results_folder):
        for file in os.listdir(save_folder):
            if '.json' in file:
                raise ValueError('This experiment name is already taken. Choose another one.')
            
    return save_folder + '/'



def concatenate_images(images, target=True):
    """
    Concatenate a reference image, target image, and list of images (neighbors of
    reference image) into one single image for easy visual comparison.

    Parameters
    ----------
    images : tuple of PIL image
        The images, as returned by Experiment.get_neighbors_of_query().
    target : bool, optional
        Whether to add the target image to the visualization. The default is True.

    Returns
    -------
    PIL image
        The concatenation (the ref is on the first line)

    """
    
    # Resize all images to 300x300
    neighbor_images = [image.resize((300,300), Image.BICUBIC) for image in images[1]]
    ref_image = images[0].resize((300,300), Image.BICUBIC)
    if target:
        target_image = images[2].resize((300,300), Image.BICUBIC)
    
    Nlines = len(neighbor_images) // 3 + 1
    if len(neighbor_images) % 3 != 0:
        Nlines += 1
        
    Ncols = 3 if len(neighbor_images) >= 3 else len(neighbor_images)
    if target:
        Ncols = Ncols if Ncols == 3 else 2
        
    big_offset = 40
    small_offset = 10
    
    final_image = np.zeros((300*Nlines + big_offset + (Nlines-2)*small_offset,
                            300*Ncols + (Ncols-1)*small_offset, 3), dtype='uint8')
    
    if target and Ncols == 2:
        final_image = np.zeros((300*Nlines + big_offset + (Nlines-2)*small_offset,
                                300*Ncols + big_offset, 3), dtype='uint8')
    
    if target:
        start = int((final_image.shape[1] - (600+big_offset))/2)
        final_image[0:300, start:start+300, :] = np.array(ref_image)
        start += 300 + big_offset
        final_image[0:300, start:start+300, :] = np.array(target_image)
    else:
        start = int((final_image.shape[1] - 300)/2)
        final_image[0:300, start:start+300, :] = np.array(ref_image)
    
    start_i = 300 + big_offset
    if Ncols <= 2:
        start_j = int((final_image.shape[1] - (len(neighbor_images)*(300+small_offset)-small_offset))/2)
    else:
        start_j = 0
    index = 0
    for i in range(1, Nlines):
        for j in range(Ncols):
            if index < len(neighbor_images):
                final_image[start_i:start_i+300, start_j:start_j+300, :] = np.array(neighbor_images[index])
                index += 1
                start_j += 300 + small_offset
        if Ncols <= 2:
            start_j = int((final_image.shape[1] - (len(neighbor_images)*(300+small_offset)-small_offset))/2)
        else:
            start_j = 0
        start_i += 300 + small_offset
            
    return Image.fromarray(final_image)
    
