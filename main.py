#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""

import faiss
from helpers import utils
import time

method = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_templates'
dataset_retrieval = 'Kaggle_memes'

t0 = time.time()

features, mapping = utils.combine_features(method, dataset)
features_search, mapping_search = utils.load_features(method, dataset_retrieval)

t1 = time.time()

print(f'Time for loading data : {t1 - t0:.2f} s', flush=True)

# res = faiss.StandardGpuResources()  # use a single GPU

index = faiss.IndexFlat(features.shape[1], metric=faiss.METRIC_JensenShannon)
# index = faiss.IndexFlatL2(features.shape[1])
# index = faiss.index_cpu_to_gpu(res, 0, index)

t2 = time.time()

print(f'Time for creating index: {t2 - t1:.2f} s', flush=True)

index.add(features)

t3 = time.time()

print(f'Time for adding to index : {t3 - t2:.2f} s', flush=True)

k = 1
D, I = index.search(features_search, k)

t4 = time.time() 

print(f'Time for search : {t4 - t3:.2f} s', flush=True)

print(f'\nTotal time : {t4-t0:.2f} s')