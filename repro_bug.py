#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:13:08 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
from tqdm import tqdm
import gc
import time
from helpers import utils
from PIL import Image


features_db = np.random.rand(500000, 4096).astype('float32')
features_query = np.random.rand(40000, 4096).astype('float32')

d = features_db.shape[1]
nlist = int(10*np.sqrt(features_db.shape[0]))
factory_string = f'IVF{nlist},Flat'

"""
# Works fine

indices = []
indices.append(faiss.index_factory(d, factory_string, faiss.METRIC_L2))
indices.append(faiss.index_factory(d, factory_string, faiss.METRIC_INNER_PRODUCT))

for i in tqdm(range(len(indices))):
    
    index = indices[i]
    index = faiss.index_cpu_to_all_gpus(index)
    
    index.train(features_db)
    index.add(features_db)
    index.nprobe = 1000
    
    D, I = index.search(features_query, 1)
"""

index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)
index = faiss.index_cpu_to_all_gpus(index)
    
index.train(features_db)
index.add(features_db)
index.nprobe = 1
    
D, I = index.search(features_query, 1)
    


"""
# Memory issue
index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)
index = faiss.index_cpu_to_all_gpus(index)
index.train(features_db)
index.add(features_db)

D, I = index.search(features_query, 1)

# index.reset()
# del index

index = faiss.index_factory(d, factory_string, faiss.METRIC_INNER_PRODUCT)
index = faiss.index_cpu_to_all_gpus(index)
index.train(features_db)
index.add(features_db)

D, I = index.search(features_query, 1)
"""
#%%

METRICS = {
    'JS': faiss.METRIC_JensenShannon,
    'L2': faiss.METRIC_L2,
    'L1': faiss.METRIC_L1,
    'cosine': faiss.METRIC_INNER_PRODUCT,
    }

class Data(object):
    
    def __init__(self, features_db=None, features_query=None):
        
        self.features_db = np.random.rand(500000, 4096).astype('float32')
        self.features_query = np.random.rand(40000, 4096).astype('float32')
        self.d = self.features_db.shape[1]
        
    def get_features_db(self, metric):
        
        if metric == 'cosine':
            return utils.normalize(self.features_db)
        else:
            return self.features_db
        
    def get_features_query(self, metric):
        
        if metric == 'cosine':
            return utils.normalize(self.features_query)
        else:
            return self.features_query
        
        
        
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
    
    def __init__(self, data):
        self.data = data

        
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
            gc.collect()
        except AttributeError:
            pass
        
        if (metric not in METRICS):
            raise ValueError(f'Metric should be one of {METRICS.keys(),}')
            
        self.factory_str = factory_str
        self.metric = metric
        self.experiment_name = factory_str + '--' + metric
        self.index = faiss.index_factory(self.data.d, factory_str, METRICS[metric])
        
        
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
        
        features = self.data.get_features_db(self.metric)
            
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
        
        features_query = self.data.get_features_query(self.metric)
            
        # Do not add normalization time
        t0 = time.time()
        
        self.D, self.I = self.index.search(features_query, self.k)
        
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
        
        return 0.5
    
            
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
        
        if type(k) == list and type(probe==list):
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
    
    
    
class Experiment2(object):
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
    
    def __init__(self):
        
        self.features_db = np.random.rand(500000, 4096).astype('float32')
        self.features_query = np.random.rand(40000, 4096).astype('float32')
        self.d = self.features_db.shape[1]
        
        self.features_db_normalized = utils.normalize(self.features_db)
        self.features_query_normalized = utils.normalize(self.features_query)
        
        
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
            
        # Do not add normalization time
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
        
        return 0.5
    
            
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
        
        if type(k) == list and type(probe==list):
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
    
    
"""
t0 = time.time()
nlist = int(10*np.sqrt(500000))
nprobes = [1, 5, 10, 20, 50, 100]#, 200, 300, 400]
factory_str = ['Flat', f'IVF{nlist},Flat']
metrics = ['L2', 'cosine']
# data = Data()
experiment = Experiment2()

for string in factory_str:
    for metric in metrics:
        experiment.set_index(string, metric)
        experiment.to_gpu()
        experiment.fit(k=1, probe=nprobes)
        # experiment.fit(1)
        
print(f'Done in {time.time() - t0:.2f} s')      
"""