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

features, mapping = utils.combine_features(method, dataset)
features_search, mapping_search = utils.load_features(method, dataset_retrieval)

res = faiss.StandardGpuResources()  # use a single GPU

index = faiss.IndexFlatL2(features.shape[1])
index = faiss.index_cpu_to_gpu(res, 0, index)

t0 = time.time()

index.add(features)

t1 = time.time()

print('Time for adding to index : {t0 - t1:.2f} s', flush=True)

k = 1
D, I = index.search(features_search, k)

dt = time.time() - t1

print('Time for search : {dt - t1:.2f} s', flush=True)