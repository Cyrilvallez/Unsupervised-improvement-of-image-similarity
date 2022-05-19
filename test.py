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
import torch

algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'BSDS500_original'
# main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'
# query_dataset = 'Kaggle_memes'

features_db, mapping_db = utils.combine_features(algorithm, main_dataset,
                                                           distractor_dataset)
features_query, mapping_query = utils.load_features(algorithm, query_dataset)

d = features_db.shape[1]

device = torch.device('cuda')

features_db = torch.tensor(features_db).to(device)
features_query = torch.tensor(features_query).to(device)


nlist = int(10*np.sqrt(features_db.shape[0]))

factory_string = f'IVF{nlist},Flat'
index = faiss.index_factory(d, factory_string)
index = faiss.index_cpu_to_all_gpus(index)


t0 = time.time()

index.train(features_db)
index.add(features_db)

t1 = time.time()

print(f'Training time : {t1 - t0:.2f} s', flush=True)

D1,I1 = index.search(features_query, 1)

t2 = time.time()

print(f'Searching time : {t2 - t1:.2f} s', flush=True)




