#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:46:20 2022

@author: cyrilvallez
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import extractor.neural as neural
import extractor.perceptual as perceptual

def extract_and_save_neural(model_name, dataset, dataset_and_model_name, batch_size=256,
                              workers=4, device='cuda'):
    """
    Compute and save features of a neural model to file.

    Parameters
    ----------
    model_name : string
        The name of the model for features extraction.
    dataset : Custom PyTorch Dataset
        The dataset containing the images for which to extract features.
    dataset_and_model_name : string
        Name of the dataset and model used (for the filenames).
    batch_size : float, optional
        The batch size for the dataloader. The default is 256.
    workers : int, optional
        The number of workers for data loading. The default is 4.
    device : string
        The device on which to perform computations. The default is 'cuda'.

    Returns
    -------
    None.

    """
    
    if (model_name not in neural.MODEL_LOADER.keys()):
        raise ValueError(f'Model must be one of {neural.MODEL_LOADER.keys()}.')
        
    if (device != 'cuda' and device != 'cpu'):
        raise ValueError('Device must be either `cuda` or `cpu`.')
        
    model = neural.MODEL_LOADER[model_name](device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    
    start = 0

    for images, names in tqdm(dataloader):
        
        images = images.to(device)
        
        with torch.no_grad():
            feats = model(images).cpu().numpy()
        
        if start==0:
            features = np.empty((len(dataset), len(feats[0])))
            indices_to_names = np.empty(len(dataset), dtype=object)
            
        N = len(names)
            
        # First column is the identifier of the image
        features[start:start+N, :] = feats
        indices_to_names[start:start+N] = names
        
        start += N
    
    np.save(dataset_and_model_name + '_features.npy', features)
    np.save(dataset_and_model_name + '_map_to_names.npy', indices_to_names)
    
    
    
def extract_and_save_perceptual(algorithm, dataset, dataset_and_algo_name,
                                hash_size=8):
    """
    Compute and save hashes of a perceptual algorithm to file.

    Parameters
    ----------
    algorithm : string
        The name of the algorithm.
    dataset : Custom PyTorch Dataset
        The dataset containing the images for which to extract features.
    dataset_and_algo_name : string
        Name of the dataset and algorithm used (for the filenames).
    hash_size : int
    The hash size to use for the algorithm. Note that this is squared, thus giving
    8 is equivalent to a hash length of 8**2=64. The default is 8.
    
    Returns
    -------
    None.

    """
    
    if (algorithm not in perceptual.NAME_TO_ALGO.keys()):
        raise ValueError(f'Algorithm must be one of {perceptual.NAME_TO_ALGO.keys()}.')
    
    algorithm = perceptual.NAME_TO_ALGO[algorithm]
    
    features = np.empty((len(dataset), hash_size**2))
    indices_to_names = np.empty(len(dataset), dtype=object)
    
    for i in tqdm(range(len(dataset))):
        image, name = dataset[i]
        features[i,:] = algorithm(image, hash_size=hash_size)
        indices_to_names[i] = name
    
    np.save(dataset_and_algo_name + '_features.npy', features)
    np.save(dataset_and_algo_name + '_map_to_names.npy', indices_to_names)
    