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
from extractor.datasets import collate

def extract_neural(model, dataset, batch_size=256, workers=8, device='cuda'):
    """
    Compute features of a neural model.

    Parameters
    ----------
    model : string, or torch nn.Module
        The name of the model for features extraction, or the model itself.
    dataset : Custom PyTorch Dataset
        The dataset containing the images for which to extract features.
    batch_size : float, optional
        The batch size for the dataloader. The default is 256.
    workers : int, optional
        The number of workers for data loading. The default is 4.
    device : string
        The device on which to perform computations. The default is 'cuda'.

    Raises
    ------
    ValueError
        If invalid value for model or device.

    Returns
    -------
    features : Numpy array
        The features.
    indices_to_names : Numpy array
        A mapping from indices to names of images.

    """
    
    if type(model) == str:
        if (model not in neural.MODEL_LOADER.keys()):
            raise ValueError(f'Model must be one of {*neural.MODEL_LOADER.keys(),}.')
        
    if (device != 'cuda' and device != 'cpu'):
        raise ValueError('Device must be either `cuda` or `cpu`.')
        
    if type(model) == str:
        model = neural.MODEL_LOADER[model](device)
        
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    
    start = 0

    for images, names in tqdm(dataloader):
        
        images = images.to(device)
        
        with torch.no_grad():
            feats = model(images).cpu().numpy()
        
        if start==0:
            features = np.empty((len(dataset), len(feats[0])), dtype='float32')
            indices_to_names = np.empty(len(dataset), dtype=object)
            
        N = len(names)
            
        # First column is the identifier of the image
        features[start:start+N, :] = feats
        indices_to_names[start:start+N] = names
        
        start += N
    
    return features, indices_to_names



def extract_and_save_neural(model, dataset, name=None, batch_size=256,
                              workers=8, device='cuda'):
    """
    Compute and save features of a neural model to file.

    Parameters
    ----------
    model : string, or torch nn.Module
        The name of the model for features extraction, or the model itself.
    dataset : Custom PyTorch Dataset
        The dataset containing the images for which to extract features.
    name : str, optional
        A name for the model if `model` is a torch.nn.Module. Otherwise this is
        ignored. The default is None.
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
    
    features, indices_to_names = extract_neural(model, dataset, batch_size=batch_size,
                                                workers=workers, device=device)
    
    if name is None:
        dataset_and_model_name = dataset.name + '-' + '_'.join(model.split(' '))
    else:
        dataset_and_model_name = dataset.name + '-' + '_'.join(name.split(' '))
    
    np.save('Features/' + dataset_and_model_name + '_features.npy', features)
    np.save('Features/' + dataset_and_model_name + '_map_to_names.npy', indices_to_names)
    
    
    
def extract_perceptual(algorithm, dataset, hash_size=8, batch_size=2048,
                       workers=8):
    """
    Compute hashes of a perceptual algorithm.

    Parameters
    ----------
    algorithm : string
        The name of the algorithm.
    dataset : Custom PyTorch Dataset
        The dataset containing the images for which to extract features.
    hash_size : int
        The hash size to use for the algorithm. Note that this is squared, thus
        giving 8 is equivalent to a hash length of 8**2=64. The default is 8.
    batch_size : float, optional
        The batch size for the dataloader. The default is 2048.
    workers : int, optional
        The number of workers for data loading. The default is 8.

    Raises
    ------
    ValueError
        If invalid value for model or device.

    Returns
    -------
    features : Numpy array
        The hashes.
    indices_to_names : Numpy array
        A mapping from indices to names of images.

    """
    
    if (algorithm not in perceptual.NAME_TO_ALGO.keys()):
        raise ValueError(f'Algorithm must be one of {*perceptual.NAME_TO_ALGO.keys(),}.')
    
    algorithm = perceptual.NAME_TO_ALGO[algorithm]
    
    features = np.empty((len(dataset), hash_size**2), dtype='uint8')
    indices_to_names = np.empty(len(dataset), dtype=object)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, collate_fn=collate)
    
    index = 0
    
    for images, names in tqdm(dataloader):
    
        for image, name in zip(images, names):
            features[index,:] = algorithm(image, hash_size=hash_size)
            indices_to_names[index] = name
            index += 1
            
    return features, indices_to_names


    
def extract_and_save_perceptual(algorithm, dataset, hash_size=8,
                                batch_size=2048, workers=8):
    """
    Compute and save hashes of a perceptual algorithm to file.

    Parameters
    ----------
    algorithm : string
        The name of the algorithm.
    dataset : Custom PyTorch Dataset
        The dataset containing the images for which to extract features.
    hash_size : int
        The hash size to use for the algorithm. Note that this is squared, thus
        giving 8 is equivalent to a hash length of 8**2=64. The default is 8.
    batch_size : float, optional
        The batch size for the dataloader. The default is 2048.
    workers : int, optional
        The number of workers for data loading. The default is 8.
    
    Returns
    -------
    None.

    """
    
    features, indices_to_names = extract_perceptual(algorithm, dataset, hash_size=hash_size,
                                                    batch_size=batch_size, workers=workers)
    
    dataset_and_algo_name = dataset.name + '-' + '_'.join(algorithm.split(' ')) \
        + f'_{hash_size**2}_bits'
    
    np.save('Features/' + dataset_and_algo_name + '_features.npy', features)
    np.save('Features/' + dataset_and_algo_name + '_map_to_names.npy', indices_to_names)
    
    