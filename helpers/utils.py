#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:02:28 2022

@author: cyrilvallez
"""
import numpy as np
import faiss

import extractor.datasets as datasets
from extractor.neural import MODEL_LOADER
from extractor.perceptual import NAME_TO_ALGO

DATASET_DIMS = {
    'Kaggle_templates': 250,
    'Kaggle_memes' : 43660,
    'BSDS500_original' : 500,
    'BSDS500_attacks' : 11600,
    'Flickr500K': 500000,
    }


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
    
    if (method_name not in MODEL_LOADER.keys() and method_name not in NAME_TO_ALGO.keys()):
        raise ValueError('Method name must be one of {MODEL_LOADER.keys()} or {NAME_TO_ALGO.keys()}.')
    
    if dataset_name not in datasets.VALID_DATASET_NAMES:
        raise ValueError('The dataset name must be one of {VALID_DATASET_NAMES}.')
        
    path = 'Features/' + dataset_name + '-' + '_'.join(method_name.split(' '))

    features = np.load(path + '_features.npy').astype('float32')
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
    features = np.empty((N1 + N2, M), dtype='float32')
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

