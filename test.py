#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
import time

from helpers import utils
"""
algorithm = 'Dhash 64 bits'
main_dataset = 'BSDS500_original'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'

features_db, mapping_db = utils.combine_features(algorithm, main_dataset, distractor_dataset)
features_query, mapping_query = utils.load_features(algorithm, query_dataset)

res = faiss.StandardGpuResources()

index = faiss.GpuIndexBinaryFlat(res, features_db.shape[1]*8)
# index = faiss.index_cpu_to_all_gpus(index)
# index = faiss.index_cpu_to_gpu(res, 0, index)

t0 = time.time()

index.add(features_db)
D, I = index.search(features_query, 10)

dt = time.time() - t0

recall, _ = utils.recall(I, mapping_db, mapping_query)

print(f'{dt:.2f} s')
print(f'Recall : {recall:.2f}')
"""

index = faiss.GpuIndexBinaryFlat(32)