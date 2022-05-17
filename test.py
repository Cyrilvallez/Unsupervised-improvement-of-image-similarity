#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
from helpers import utils
import time

method = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_templates'
dataset_retrieval = 'Kaggle_memes'

features_db, mapping_db = utils.combine_features(method, dataset)
features_query, mapping_query = utils.load_features(method, dataset_retrieval)

d = features_db.shape[1]

res = faiss.StandardGpuResources()  

# coarse = faiss.IndexFlat(d)
# coarse.metric_type = faiss.METRIC_JensenShannon
# coarse = faiss.index_cpu_to_gpu(res, 0, coarse)

nlist = int(10*np.sqrt(features_db.shape[0]))
# index = faiss.IndexIVFFlat(coarse, d, nlist)

factory_string = f'IVF{nlist}'
index1 = faiss.index_factory(d, factory_string)
index1 = faiss.index_cpu_to_gpu(res, 0, index1)

t0 = time.time()

index1.train(features_db)
index1.add(features_db)

t1 = time.time()

print(f'Training time : {t1 - t0:.2f} s', flush=True)

D1,I1 = index1.search(features_query, 1)

t2 = time.time()

print(f'Searching time : {t2 - t1:.2f} s', flush=True)

# recall, _ = utils.recall(I, mapping_db, mapping_query)

# print(f'Recall : {recall:.2f}')

del index1

index2 = faiss.index_factory(d, factory_string)

index_ivf = faiss.extract_index_ivf(index2)
clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
index_ivf.clustering_index = clustering_index

# index2 = faiss.index_cpu_to_all_gpus(index2)


t3 = time.time()
index2.train(features_db)
index2.add(features_db)
t4 = time.time()

print(f'Training time : {t4 - t3:.2f} s', flush=True)

D2,I2 = index2.search(features_query, 1)

t5 = time.time()

print(f'Searching time : {t5 - t4:.2f} s', flush=True)

assert ((D1==D2).all())
assert ((I1==I2).all())
print('Everything is same')

#%%


# factory_str = 'Flat'
# index = faiss.index_factory(12, factory_str, faiss.METRIC_JensenShannon)
# print(index.metric_type)


# from helpers import utils

# method = 'SimCLR v2 ResNet50 2x'
# dataset = 'Kaggle_templates'
# dataset_retrieval = 'Kaggle_memes'

# features_db, mapping_db = utils.combine_features(method, dataset)
# features_query, mapping_query = utils.load_features(method, dataset_retrieval)

# d = features_db.shape[1]

# coarse = faiss.IndexFlatL2(d)
# nlist = 100
# index = faiss.IndexIVFFlat(coarse, d, nlist)

# index.train(features_db)


