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

index = faiss.IndexFlat(d)
index.metric_type = faiss.METRIC_L1
coarse = faiss.index_cpu_to_gpu(res, 0, index)

nlist = int(10*np.sqrt(features_db.shape[0]))
index = faiss.IndexIVFFlat(coarse, d, nlist)
index = faiss.index_cpu_to_gpu(res, 0, index)

t0 = time.time()

index.train(features_db)

t1 = time.time()

print(f'Training time : {t1 - t0:.2f} s', flush=True)

D,I = index.search(features_query, 1)

t2 = time.time()

print(f'Searching time : {t2 - t1:.2f} s', flush=True)

recall, _ = utils.recall(I, mapping_db, mapping_query)

print(f'Recall : {recall:.2f}')