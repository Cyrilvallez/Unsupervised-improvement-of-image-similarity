#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""


import numpy as np
import faiss

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

features_db, mapping_db = utils.combine_features(algorithm, main_dataset,
                                                           distractor_dataset)
features_query, mapping_query = utils.load_features(algorithm, query_dataset)


nlist = int(10*np.sqrt(500000))
factory_str = f'IndexIVF{nlist},Flat'
# nprobes = [1, 5, 10, 20, 50, 100, 200, 300, 400]

# filename = save_folder + 'results.json'
# ex.compare_nprobe_IVF(nlist, nprobes, algorithm, main_dataset, query_dataset,
                        # distractor_dataset, filename, k=1)
                        
index = faiss.index_factory(features_db.shape[1], factory_str)
index = faiss.index_cpu_to_all_gpus(index)

index.train(features_db)
index.add(features_db)

D,I = index.search(features_query, 1)

recall = utils.recall(I, mapping_db, mapping_query)

print(f'Recall : {recall}')

