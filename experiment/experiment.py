#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:15:56 2022

@author: cyrilvallez
"""

import faiss
import time
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES=True

from helpers import utils

METRICS = {
    'JS': faiss.METRIC_JensenShannon,
    'L2': faiss.METRIC_L2,
    'L1': faiss.METRIC_L1,
    'cosine': faiss.METRIC_INNER_PRODUCT,
    }

class Experiment(object):
    """
    Class representing data alongside index method for easier manipulation.

    Parameters
    ----------
    algorithm : string
        The name of the algorithm used to extract the features.
    main_dataset : string
        Name of the main dataset from which the features were extracted.
    query_dataset : string
        Name of the dataset from which the query features were extracted.
    distractor_dataset : string, optional
        Name of the dataset used as distractor features (images) for the 
        database.

    """
    
    def __init__(self, algorithm, main_dataset, query_dataset,
                 distractor_dataset='Flickr500K'):
        
        self.algorithm = algorithm
        self.binary = True if 'bits' in self.algorithm else False
        self.main_dataset = main_dataset
        self.query_dataset = query_dataset
        self.distractor_dataset = distractor_dataset
        
        self.features_db, self.mapping_db = utils.combine_features(algorithm, main_dataset,
                                                                   distractor_dataset)
        self.features_query, self.mapping_query = utils.load_features(algorithm, query_dataset)
        if not self.binary:
            self.features_db_normalized = utils.normalize(self.features_db)
            self.features_query_normalized = utils.normalize(self.features_query)
        
        # For binary indices, data is represented as array of bytes
        if self.binary:
            self.d = self.features_db.shape[1]*8
        else:
            self.d = self.features_db.shape[1]
        
        self.identifiers_query = np.array([name.rsplit('/', 1)[1].split('_', 1)[0] \
                                           for name in self.mapping_query])
        self.identifiers_db = np.array([name.rsplit('/', 1)[1].rsplit('.', 1)[0] \
                                        for name in self.mapping_db])
            
        
    def set_index(self, factory_str, metric='cosine'):
        """
        Set the current index to a new one. Allows not to reload data but still
        performing new experiments. This is not set in the contructor because
        we usually change it inside loops.

        Parameters
        ----------
        factory_str : string
            The factory string for the new index.
        metric : string, optional
            String identifier of the metric to be used. Ignored if `binary`
            is set to True.The default is 'cosine'.

        Returns
        -------
        None.

        """
        
        try:
            # Free memory if an index already exists
            self.index.reset()
            del self.index
        except AttributeError:
            pass
        
        if (metric not in METRICS and not self.binary):
            raise ValueError(f'Metric should be one of {*METRICS.keys(),}')
            
        self.factory_str = factory_str

        if not self.binary:
            self.index = faiss.index_factory(self.d, factory_str, METRICS[metric])
            self.metric = metric
        else:
            self.index = faiss.index_binary_factory(self.d, factory_str)
            self.metric = 'Hamming'
            
        self.experiment_name = factory_str + '--' + metric
        
        
    def to_gpu(self):
        """
        Put the index on GPU, if it is not a binary index (since the equivalent
        for binary indices is not supported in faiss).

        Returns
        -------
        None.

        """
        if not self.binary:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        
    def train(self):
        """
        Train the index and add database vectors to it.

        Returns
        -------
        None.

        """
            
        # Do not add normalization time
        t0 = time.time()

        if self.metric == 'cosine':
            self.index.train(self.features_db_normalized)
            self.index.add(self.features_db_normalized)
        else:
            self.index.train(self.features_db)
            self.index.add(self.features_db)
        
        self.time_training = time.time() - t0
        
        
    def search(self, k=None, probe=None, batch_size=1000):
        """
        Perform a search given k-nearest neigbors and a probe for indices
        supporting it.

        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors. If omitted, will default to the 
            actual value of the class which is 1 at construction time. 
            The default is None.
        probe : int, optional
            The number of clusters to visit for IVF indices. If omitted, will default to the 
            actual value of the index which is 1 at construction time. The default is None.
        batch_size : int, optional
            The batch size in case we need to divide the query into batches to
            avoir memory overflow. The default is 1000.

        Returns
        -------
        None.

        """
        
        if k is not None:
            self.k = k
        
        if probe is not None:
            self.index.nprobe = probe
            
        t0 = time.time()
        
        if 'IVF' in self.factory_str and not self.index_binary:
            self.search_in_batch(batch_size)
        else:
            if self.metric == 'cosine':
                self.D, self.I = self.index.search(self.features_query_normalized, self.k)
            else:
                self.D, self.I = self.index.search(self.features_query, self.k)
                
        self.time_searching = time.time() - t0
        
        
    def search_in_batch(self, batch_size=1000):
        """
        Perform a search of the index in batch (to avoid possible memory overflows
        while searching with an IVF index, see https://github.com/facebookresearch/faiss/issues/2338).

        Parameters
        ----------
        batch_size : int, optional
            The batch size. The default is 1000.

        Returns
        -------
        None.

        """
        
        length = len(self.features_query)
        N_iter = length//batch_size
        
        self.D = np.zeros((length, self.k), dtype='float32')
        self.I = np.zeros((length, self.k), dtype='int64')
        
        N = 0
        for i in range(N_iter):
            if self.metric == 'cosine':
                D, I = self.index.search(
                    self.features_query_normalized[N:N+batch_size, :], self.k)
            else:
                D, I = self.index.search(
                    self.features_query[N:N+batch_size, :], self.k)
                
            self.D[N:N+batch_size, :] = D
            self.I[N:N+batch_size, :] = I
            N += batch_size
            
        # Last iteration (will usually not be of the same size as the others)
        if self.metric == 'cosine':
            D, I = self.index.search(
                self.features_query_normalized[N:, :], self.k)
        else:
            D, I = self.index.search(
                self.features_query[N:, :], self.k)
            
        self.D[N:, :] = D
        self.I[N:, :] = I
        
        
    def recall(self):
        """
        Compute the recall (can only be called after search has been performed 
        at least 1 time).

        Raises
        ------
        AttributeError
            If search has not yet been called..

        Returns
        -------
        recall : float
            Recall of the experiment.

        """
        
        try:
            shape = self.I.shape
            names = self.identifiers_db[self.I.flatten()].reshape(shape)
            
            query_identifiers = np.expand_dims(self.identifiers_query, axis=1)
            
            correct = (names == query_identifiers).sum(axis=1)
            recall = correct.sum()/len(correct)
            
        except AttributeError:
            raise AttributeError('Please call `search` before asking for recall')
            
        return recall, correct
    
            
    def get_neighbors_of_query(self, query_index, target=True):
        """
        Find nearest neighbors of a given query index as PIL images.

        Parameters
        ----------
        query_index : int
            Index of the query.
        target : bool, optional
            Whether to add the target image to the output. The default is True.

        Returns
        -------
        ref_image : PIL image
            Image corresponding to the index of the query.
        neighbors : list
            List of PIL images corresponding to nearest neighbors.
        target_image : PIL image
            The image supposed to correspond to the given query.

        """
        
        ref_image = Image.open(self.mapping_query[query_index]).convert('RGB')
        if target: 
            target_index = np.argwhere(self.identifiers_db == self.identifiers_query[query_index])
            assert (target_index.shape == (1,1)), 'More than one target for this query'
            target_index = target_index[0][0]
            target_image = Image.open(self.mapping_db[target_index]).convert('RGB')
            
        neighbors = []
        
        for image_index in self.I[query_index, :]:
            neighbors.append(Image.open(self.mapping_db[image_index]).convert('RGB'))
            
        if target:
            return (ref_image, neighbors, target_image)
        else:
            return (ref_image, neighbors)
    
    
    def fit(self, k, probe=None):
        """
        Combine train, search and recall in one handy function.

        Parameters
        ----------
        k : int or list
            The number of nearest neighbors.
        probe : int, optional
            The number of clusters to visit for IVF indices. If omitted, will default to the 
            actual value of the index which is 1 at construction time. The default is None.

        Returns
        -------
        result : dictionary
            Contains recall and times for the experiment.

        """
            
        self.train()
        
        if (type(k) == list and type(probe) == list):
            raise ValueError('Cannot set both k and probe to lists.')
        
        elif type(k) == list:
            recall = []
            searching_time = []
            for k_ in k:
                self.search(k=k_, probe=probe)
                recall.append(self.recall()[0])
                searching_time.append(self.time_searching)
        
        elif type(probe) == list:
            recall = []
            searching_time = []
            for probe_ in probe:
                self.search(k=k, probe=probe_)
                recall.append(self.recall()[0])
                searching_time.append(self.time_searching)
                
        else:
            self.search(k=k, probe=probe)
            recall = self.recall()[0]
            searching_time = self.time_searching
        
        result = {
            'recall': recall,
            'training_time': self.time_training,
            'searching_time': searching_time,
            'k': k,
                  }
        
        if probe is not None:
            result['nprobe'] = probe
        
        return result
    
         

def compare_metrics_Flat(metrics, algorithm, main_dataset, query_dataset,
                         distractor_dataset, filename, k=1):
    """
    Compare the performances of different metrics when using a Flat index
    (brute-force).

    Parameters
    ----------
    metrics : list
        String identifier of the metrics to be used. Ignored if `binary`
        is set to True.
    algorithm : string
        The name of the algorithm used to extract the features.
    main_dataset : string
        Name of the main dataset from which the features were extracted.
    query_dataset : string
        Name of the dataset from which the query features were extracted.
    distractor_dataset : string, optional
        Name of the dataset used as distractor features (images) for the 
        database.
    filename : string
        Filename to save the results.
    k : int, optional
        Number of nearest neighbors for the search.

    Returns
    -------
    None.

    """

    experiment = Experiment(algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset)
    
    if not experiment.binary:
        factory_str = 'Flat'
    else:
        factory_str = 'BFlat'

    result = {}
    
    for metric in metrics:
        experiment.set_index(factory_str, metric=metric)
        experiment.to_gpu()
        result[experiment.experiment_name] = experiment.fit(k=k)
        
    utils.save_dictionary(result, filename)
    
    
    
def compare_k_Flat(ks, algorithm, main_dataset, query_dataset,
                         distractor_dataset, filename):
    """
    Compare the performances of different k when using a Flat index
    (brute-force).

    Parameters
    ----------
    ks : list
        The different k to be used.
    algorithm : string
        The name of the algorithm used to extract the features.
    main_dataset : string
        Name of the main dataset from which the features were extracted.
    query_dataset : string
        Name of the dataset from which the query features were extracted.
    distractor_dataset : string, optional
        Name of the dataset used as distractor features (images) for the 
        database.
    filename : string
        Filename to save the results.

    Returns
    -------
    None.

    """

    experiment = Experiment(algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset)
    
    if not experiment.binary:
        factory_str = 'Flat'
    else:
        factory_str = 'BFlat'
        
    metrics = ['L2', 'cosine']

    result = {}
    
    for metric in metrics:
        experiment.set_index(factory_str, metric=metric)
        experiment.to_gpu()
        result[experiment.experiment_name] = experiment.fit(k=ks)
        
    utils.save_dictionary(result, filename)
    
    
def compare_nprobe_IVF(nlist, nprobes, algorithm, main_dataset, query_dataset,
                         distractor_dataset, filename, k=1):
    """
    Compare the performaces of IVF vs Flat for cosine and L2 metrics, for
    different nprobes.

    Parameters
    ----------
    nlist : int
        The number of inverted lists.
    nprobes : list
        The different nprobes.
    algorithm : string
        The name of the algorithm used to extract the features.
    main_dataset : string
        Name of the main dataset from which the features were extracted.
    query_dataset : string
        Name of the dataset from which the query features were extracted.
    distractor_dataset : string, optional
        Name of the dataset used as distractor features (images) for the 
        database.
    filename : string
        Filename to save the results.
    k : int, optional
        Number of nearest neighbors for the search.
    

    Returns
    -------
    None.

    """
    
    assert(max(nprobes) <= nlist)

    experiment = Experiment(algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset)
    
    if not experiment.binary:
        factory_str = ['Flat', f'IVF{nlist},Flat']
    else:
        factory_str = ['BFlat', f'BIVF{nlist},Flat']
        
    metrics = ['L2', 'cosine']

    result = {}
    
    for metric in tqdm(metrics):
        experiment.set_index(factory_str[0], metric=metric)
        experiment.to_gpu()
        result[experiment.experiment_name] = experiment.fit(k=k)
        
    for metric in tqdm(metrics):
        experiment.set_index(factory_str[1], metric=metric)
        experiment.to_gpu()
        result[experiment.experiment_name] = experiment.fit(k=k, probe=nprobes)
            
    utils.save_dictionary(result, filename)
    