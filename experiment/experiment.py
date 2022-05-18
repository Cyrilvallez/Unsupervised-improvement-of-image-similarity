#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:15:56 2022

@author: cyrilvallez
"""

import faiss
import time
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

from helpers import utils


class Experiment():
    """
    Class representing data alongside index method for easier manipulation.

    Parameters
    ----------
    factory_str : string
        The factory string for the new index.
    algorithm : string
        The name of the algorithm used to extract the features.
    main_dataset : string
        Name of the main dataset from which the features were extracted.
    query_dataset : string
        Name of the dataset from which the query features were extracted.
    distractor_dataset : string, optional
        Name of the dataset used as distractor features (images) for the 
        database.
    metric : string, optional
        String identifier of the metric to be used for the index. The default
        is 'cosine'.
    k : int, optional
        Number of nearest neighbors for the search.


    """
    
    def __init__(self, factory_str, algorithm, main_dataset, query_dataset,
                 distractor_dataset='Flickr500K', metric='cosine', k=1):
        
        if (metric not in METRICS):
            raise ValueError(f'Metric should be one of {METRICS.keys(),}')
        
        self.factory_str = factory_str
        self.algorithm = algorithm
        self.main_dataset = main_dataset
        self.query_dataset = query_dataset
        self.distractor_dataset = distractor_dataset
        
        self.features_db, self.mapping_db = utils.combine_features(algorithm, main_dataset,
                                                                   distractor_dataset)
        self.features_query, self.mapping_query = utils.load_features(algorithm, query_dataset)
        self.d = self.features_db.shape[1]
        self.k = k
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
        
    def set_index(self, factory_str, metric='cosine'):
        """
        Change the current index to a new one. Allows not to reload data but still
        performing new experiments.

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
        self.factory_str = factory_str
        self.metric = metric
        self.experiment_name = factory_str + '--' + metric
        self.index = faiss.index_factory(self.d, factory_str, METRICS[metric])
        
    def train(self):
        """
        Train the index and add database vectors to it.

        Returns
        -------
        None.

        """
        
        if self.metric == 'cosine':
            features = utils.normalize(self.features_db)
        else:
            features = self.features_db
            
        # Do not add normalization time
        t0 = time.time()

        self.index.train(features)
        self.index.add(features)
        
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
        
        if self.metric == 'cosine':
            features = utils.normalize(self.features_query)
        else:
            features = self.features_query
            
        # Do not add normalization time
        t0 = time.time()
        
        self.D, self.I = self.index.search(features, self.k)
        
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
        neighbors : list
            List of PIL images corresponding to nearest neighbors.

        """
        
        neighbors = []
        
        for image_index in self.I[query_index, :]:
            neighbors.append(Image.open(self.mapping_query[image_index]))
            
        return neighbors
    
    def fit(self, k=None, probe=None):
        """
        Combine train, search and recall in one handy function.

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
        result : dictionary
            Contains recall and times for the experiment.

        """
        
        self.train()
        self.search(k=k, probe=probe)
        recall = self.recall()
        
        result = {
            f'recall@{self.k}': recall,
            'training_time': self.time_training,
            'searching_time': self.time_searching,
            'k': self.k,
                  }
        
        # Add the nprobe value to the result if the given index has it
        try:
            result['nprobe'] = self.index.nprobe
        except AttributeError:
            pass
        
        
        return result
    
    def new_search(self, k=None, probe=None):
        """
        Combine search and recall in one function (thus new search is performed
        without retraining the index).

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
        result : dictionary
            Contains recall and times for the experiment.

        """
        
        self.search(k=k, probe=probe)
        recall = self.recall()
        
        result = {
            f'recall': recall,
            'training_time': self.time_training,
            'searching_time': self.time_searching,
            'k': self.k,
                  }
        
        # Add the nprobe value to the result if the given index has it
        try:
            result['nprobe'] = self.index.nprobe
        except AttributeError:
            pass
            
        return result
        
        


METRICS = {
    'JS': faiss.METRIC_JensenShannon,
    'L2': faiss.METRIC_L2,
    'L1': faiss.METRIC_L1,
    'cosine': faiss.METRIC_INNER_PRODUCT,
    }


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
    
    if filename is None:
        filename = 'Results/'
    
    factory_str = 'Flat'

    experiment = Experiment(factory_str, algorithm, main_dataset, query_dataset,
                            distractor_dataset=distractor_dataset, metric=metrics[0],
                            k=k)

    result = {}

    # experiment.to_gpu()
    result[experiment.experiment_name] = experiment.fit()
    
    for metric in metrics[1:]:
        experiment.set_index(factory_str, metric=metric)
        # experiment.to_gpu()
        result[experiment.experiment_name] = experiment.new_search()
        
    utils.save_dictionary(result, filename)
    