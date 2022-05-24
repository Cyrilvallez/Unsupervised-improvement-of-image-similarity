#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:15:56 2022

@author: cyrilvallez
"""

import faiss
import time
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
        self.main_dataset = main_dataset
        self.query_dataset = query_dataset
        self.distractor_dataset = distractor_dataset
        
        self.features_db, self.mapping_db = utils.combine_features(algorithm, main_dataset,
                                                                   distractor_dataset)
        self.features_query, self.mapping_query = utils.load_features(algorithm, query_dataset)
        
        self.features_db_normalized = utils.normalize(self.features_db)
        self.features_query_normalized = utils.normalize(self.features_query)
        self.d = self.features_db.shape[1]
        
        
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
            String identifier of the metric to be used. The default is 'cosine'.

        Returns
        -------
        None.

        """
        
        try:
            # This is needed to free the memory of current index and
            # not crach the process
            self.index.reset()
            del self.index
        except AttributeError:
            pass
        
        if (metric not in METRICS):
            raise ValueError(f'Metric should be one of {METRICS.keys(),}')
            
        self.factory_str = factory_str
        self.metric = metric
        self.experiment_name = factory_str + '--' + metric
        self.index = faiss.index_factory(self.d, factory_str, METRICS[metric])
        
        
    def to_gpu(self):
        """
        Put the index on GPU.

        Returns
        -------
        None.

        """
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
        
        
    def search(self, k=None, probe=None):
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

        Returns
        -------
        None.

        """
        
        if k is not None:
            self.k = k
        
        if probe is not None:
            self.index.nprobe = probe
            
        t0 = time.time()
        
        if self.metric == 'cosine':
            self.D, self.I = self.index.search(self.features_query_normalized, self.k)
        else:
            self.D, self.I = self.index.search(self.features_query, self.k)
        
        self.time_searching = time.time() - t0
        
        
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
            recall, _ = utils.recall(self.I, self.mapping_db, self.mapping_query)
        except AttributeError:
            raise AttributeError('Please call `search` before asking for recall')
            
        return recall
    
            
    def get_neighbors_of_query(self, query_index):
        """
        Find nearest neighbors of a given query index as PIL images.

        Parameters
        ----------
        query_index : int
            Index of the query.

        Returns
        -------
        ref_image : PIL image
            Image corresponding to the index of the query.
        neighbors : list
            List of PIL images corresponding to nearest neighbors.

        """
        
        ref_image = Image.open(self.mapping_query[query_index])
        neighbors = []
        
        for image_index in self.I[query_index, :]:
            neighbors.append(Image.open(self.mapping_db[image_index]).convert('RGB'))
            
        return ref_image, neighbors
    
    
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
                recall.append(self.recall())
                searching_time.append(self.time_searching)
        
        elif type(probe) == list:
            recall = []
            searching_time = []
            for probe_ in probe:
                self.search(k=k, probe=probe_)
                recall.append(self.recall())
                searching_time.append(self.time_searching)
                
        else:
            self.search(k=k, probe=probe)
            recall = self.recall()
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
    
    
    

def create_flat_index(d, metric='cosine'):
    
    if (metric not in METRICS):
        raise ValueError('Metric name must be one of {METRICS}.')
        
    index = faiss.IndexFlat(d)
    index.metric_type = METRICS[metric]
    
    return index


def create_IVFFlat_index(d, nlist, metric='cosine'):
    
    if (metric not in METRICS):
        raise ValueError('Metric name must be one of {METRICS}.')
        
    quantizer = create_flat_index(d, metric)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # say the coarse quantizer is deallocated by index destructor
    index.own_fields = True 
    # tell Python not to try to deallocate the pointer when exiting 
    # the function
    quantizer.this.disown()

    return index


def compare_metrics_Flat(metrics, algorithm, main_dataset, query_dataset,
                         distractor_dataset, filename, k=1):
    """
    Compare the performances of different metrics when using a Flat index
    (brute-force).

    Parameters
    ----------
    metrics : list
        String identifier of the metrics to be used.
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
    
    factory_str = 'Flat'

    experiment = Experiment(algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset)

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
    
    factory_str = 'Flat'
    metrics = ['L2', 'cosine']

    experiment = Experiment(algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset)

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
    
    factory_str = ['Flat', f'IVF{nlist},Flat']
    metrics = ['L2', 'cosine']

    experiment = Experiment(algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset)

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
    