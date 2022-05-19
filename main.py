#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""


import numpy as np
import faiss
import time
import torch

from helpers import utils
import experiment as ex

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()

algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'BSDS500_original'
# main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'
# query_dataset = 'Kaggle_memes'

# factory_str = 'Flat'
# metrics = ['L2', 'L1', 'cosine']

t0 = time.time()

features_db, mapping_db = utils.combine_features(algorithm, main_dataset,
                                                           distractor_dataset)
features_query, mapping_query = utils.load_features(algorithm, query_dataset)


device = torch.device('cuda')

# features_db = torch.tensor(features_db).to(device)
# features_query = torch.tensor(features_query).to(device)

# mapping_db = torch.tensor(mapping_db).to(device)
# mapping_query = torch.tensor(mapping_query).to(device)


nlist = int(10*np.sqrt(500000))
factory_str = f'IVF{nlist},Flat'
# nprobes = [1, 5, 10, 20, 50, 100, 200, 300, 400]

# filename = save_folder + 'results.json'
# ex.compare_nprobe_IVF(nlist, nprobes, algorithm, main_dataset, query_dataset,
                        # distractor_dataset, filename, k=1)
                        
index = faiss.index_factory(features_db.shape[1], factory_str)
# res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_all_gpus(index)
# index = faiss.index_cpu_to_gpu(res, 0, index)

index.train(features_db)
index.add(features_db)

D,I = index.search(features_query, 1)

recall = utils.recall(I, mapping_db, mapping_query)

print(f'Recall : {recall}')

print(f'Time : {time.time() - t0}')