#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
import time

import experiment as ex
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

"""
import torch
import torch.nn.functional as F
from helpers import utils

algorithm = 'SimCLR v2 ResNet50 2x'
dataset1 = 'Kaggle_memes'
dataset2 = 'Kaggle_templates'

features, _ = utils.combine_features(algorithm, dataset1, dataset2)
features = torch.tensor(features).to('cuda')

distances = F.pdist(features)
distances = distances.cpu().numpy()
np.save('distances_all_memes_L2', distances)
"""

# algorithm = 'SimCLR v2 ResNet50 2x'
algorithm = 'Dhash 64 bits'
main_dataset = 'BSDS500_original'
# main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'
# query_dataset = 'Kaggle_memes'

experiment = ex.Experiment(algorithm, main_dataset, query_dataset,
                        distractor_dataset=distractor_dataset)

print(experiment.binary)
