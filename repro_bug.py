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

features_db = np.random.rand(500000, 4096).astype('float32')
features_query = np.random.rand(40000, 4096).astype('float32')

d = features_db.shape[1]
nlist = int(10*np.sqrt(features_db.shape[0]))
factory_string = f'IVF{nlist},Flat'


# Works fine
"""
indices = []
indices.append(faiss.index_factory(d, factory_string, faiss.METRIC_L2))
indices.append(faiss.index_factory(d, factory_string, faiss.METRIC_INNER_PRODUCT))

for i in tqdm(range(len(indices))):
    
    index = indices[i]
    index = faiss.index_cpu_to_all_gpus(index)
    
    index.train(features_db)
    index.add(features_db)
    
    D, I = index.search(features_query, 1)
    
    

"""
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

class Experiment(object):
    
    def __init__(self, features_db, features_query):
        
        self.features_db = np.random.rand(500000, 4096).astype('float32')
        self.features_query = np.random.rand(40000, 4096).astype('float32')
        self.d = self.features_db.shape[1]
        
    def set_index(self, factory_str, metric):
        
        try:
            # Try to force deletion of previous index by all possible means
            self.index.reset()
            del self.index
            gc.collect()
        except AttributeError:
            pass
            
        self.index = faiss.index_factory(self.d, factory_str, metric)
        
    def to_gpu(self):
        
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        
    def fit(self):
        
        self.index.train(self.features_db)
        self.index.add(self.features_db)
        D, I = self.index.search(self.features_query, 1)
        
t0 = time.time()
        
factory_str = ['Flat', f'IVF{nlist},Flat']
metrics = [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]
experiment = Experiment(features_db, features_query)

for string in factory_str:
    for metric in metrics:
        experiment.set_index(string, metric)
        experiment.to_gpu()
        experiment.fit()
        
print(f'Done in {time.time() - t0:.2f} s')      