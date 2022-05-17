#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:15:56 2022

@author: cyrilvallez
"""

import faiss

from helpers import utils

class Experiment():
    
    def __init__(self, method, main_dataset, query_dataset, distractor_dataset='Flickr500K'):
        
        self.method = method
        self.main_dataset = main_dataset
        self.query_dataset = query_dataset
        self.distractor_dataset = distractor_dataset
        
        self.features_db, self.mapping_db = utils.combine_features(method, main_dataset,
                                                                   distractor_dataset)
        self.features_query, self.mapping_query = utils.load_features(method, query_dataset)
        
        self.d = features_db.shape[1]


METRICS = {
    'JS': faiss.METRIC_JensenShannon,
    'L2': faiss.METRIC_L2,
    'L1': faiss.METRIC_L1,
    'cosine': faiss.METRIC_INNER_PRODUCT,
    }


def create_flat_index(d, metric='L2', ressource=None):
    
    if (metric not in METRICS):
        raise ValueError('Metric name must be one of {METRICS}.')
        
    if ressource is None:
        ressource = faiss.StandardGpuResources()
        
    index = faiss.IndexFlat(d)
    index.metric_type = METRICS[metric]
    index = faiss.index_cpu_to_gpu(ressource, 0, index)
    
    return index


def create_IVFFlat_index(d, nlist, metric='L2', ressource=None):
    
    if (metric not in METRICS):
        raise ValueError('Metric name must be one of {METRICS}.')
        
    if ressource is None:
        ressource = faiss.StandardGpuResources()
        
    quantizer = create_flat_index(ressource, d, metric)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # say the coarse quantizer is deallocated by index destructor
    index.own_fields = True 
    # tell Python not to try to deallocate the pointer when exiting 
    # the function
    quantizer.this.disown()
    index = faiss.index_cpu_to_gpu(ressource, 0, index)

    return index


def IVFFLat_probes_comparison(d, nlist, nprobes, features_db, metric='L2'):
    
    index = create_IVFFlat_index(d, nlist, metric=metric)
    
    index.train(features_db)
    index.add(features_db)           
    
    
    
    