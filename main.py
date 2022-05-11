#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""

import faiss
from helpers import utils

method = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_templates'
dataset_retrieval = 'Kaggle_memes'

features, mapping = utils.combine_features(method, dataset)
features_search, mapping_search = utils.load_features(method, dataset_retrieval)

res = faiss.StandardGpuResources()  # use a single GPU

index = faiss.IndexFlatL2(features.shape[1])
index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(features)

k = 1
D, I = index.search(features_search, k)