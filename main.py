#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""

import faiss
from helpers import utils
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

method = 'SimCLR v2 ResNet50 2x'
dataset = 'BSDS500_original'
dataset_retrieval = 'BSDS500_attacks'

features_db, mapping_db = utils.combine_features(method, dataset)
features_query, mapping_query = utils.load_features(method, dataset_retrieval)
features_db_normalized = utils.normalize(features_db)
features_query_normalized = utils.normalize(features_query)

d = features_db.shape[1]
res = faiss.StandardGpuResources()  # use a single GPU
nlist = int(10*np.sqrt(features_db.shape[0]))

indices = [
    utils.create_flat_index(res, d, 'cosine'),
    utils.create_flat_index(res, d, 'L2'),
    utils.create_IVFFlat_index(res, d, nlist, 'cosine'),
    utils.create_IVFFlat_index(res, d, nlist, 'L2'),
    ]
names = [
    # 'JS Flat',
    'cosine Flat',
    'L2 Flat',
    # 'L1 Flat',
    # 'JS IVFFlat',
    'cosine IVFFlat',
    'L2 IVFFlat',
    # 'L1 IVFFlat',
    ]

recalls = []
times = []

k = 1

for i in tqdm(range(len(indices))):
    
    index = indices[i]
    
    if ' Flat' in names[i]:
        
        t0 = time.time()
    
        if 'cosine' in names[i]:
            index.add(features_db_normalized)
            D, I = index.search(features_query_normalized, k)
    
        else:
            index.add(features_db)
            D, I = index.search(features_query, k)
        
        times.append(time.time() - t0)

        recall, _ = utils.recall(I, mapping_db, mapping_query)
        recalls.append(recall)
        
    elif ' IVFFlat' in names[i]:
        
        if 'cosine' in names[i]:
            index.train(features_db_normalized)
            index.add(features_db_normalized)           
            
        else:
            index.train(features_db)
            index.add(features_db)
            
        recall_list = []
        time_list = []
        
        for nprobe in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            
            index.setNumProbes(nprobe)
            
            t0 = time.time()
            
            if 'cosine' in names[i]:
                D, I = index.search(features_query_normalized, k)
            else:
                D, I = index.search(features_query, k)
            
            recall, _ = utils.recall(I, mapping_db, mapping_query)
            recall_list.append(recall)
            time_list.append(time.time() - t0)
            
        recalls.append(recall_list)
        times.append(time_list)
    
    # Free memory while keeping list length consant
    indices[i] = 0
    del index
    
with open('recall_BSDS500.pickle', "wb") as fp:   #Pickling
    pickle.dump(recalls, fp)
    
with open('time_BSDS500.pickle', "wb") as fp:   #Pickling
    pickle.dump(times, fp)

"""
plt.figure()
for i in range(len(names)):
    if names[i] == 'JS Flat':
        plt.scatter(recalls[i], times[i], color='r', marker='*', s=100)
    if names[i] == 'cosine Flat':
        plt.scatter(recalls[i], times[i], color='b', marker='o', s=100)
    if names[i] == 'L2 Flat':
        plt.scatter(recalls[i], times[i], color='g', marker='x', s=100)
    if names[i] == 'L1 Flat':
        plt.scatter(recalls[i], times[i], color='m', marker='s', s=100)
    if names[i] == 'JS IVFFlat':
        plt.plot(recalls[i], times[i], color='r')
    if names[i] == 'cosine IVFFlat':
        plt.plot(recalls[i], times[i], color='b')
    if names[i] == 'L2 IVFFlat':
        plt.plot(recalls[i], times[i], color='g')
    if names[i] == 'L1 IVFFlat':
        plt.plot(recalls[i], times[i], color='m')

plt.xlabel('Recall@1')
plt.ylabel('Search time [s]')
plt.legend(names)
        
plt.savefig('benchmark.pdf', bbox_inches='tight')
plt.show()
"""

