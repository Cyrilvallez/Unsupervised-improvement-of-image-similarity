#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:13:08 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
from tqdm import tqdm

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
