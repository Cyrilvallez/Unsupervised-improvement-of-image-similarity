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
from helpers import configs_plot

method = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_templates'
dataset_retrieval = 'Kaggle_memes'

features_db, mapping_db = utils.combine_features(method, dataset)
features_query, mapping_query = utils.load_features(method, dataset_retrieval)

d = features_db.shape[1]

indices = []
names = []

res = faiss.StandardGpuResources()  # use a single GPU

index = faiss.IndexFlat(d)
index.metric_type = faiss.METRIC_JensenShannon

indices.append(index)
names.append('JS (CPU)') 
indices.append(faiss.IndexFlatIP(d))
names.append('cosine (CPU)')
indices.append(faiss.IndexFlatL2(d))
names.append('L2 (CPU)')
index = faiss.IndexFlatIP(d)
index = faiss.index_cpu_to_gpu(res, 0, index)
indices.append(index)
names.append('cosine (GPU)')
index = faiss.IndexFlatL2(d)
index = faiss.index_cpu_to_gpu(res, 0, index)
indices.append(index)
names.append('L2 (GPU)')

recalls = []
times = []

k = 1

for i in tqdm(range(len(indices))):
    
    index = indices[i]
    
    t0 = time.time()
    
    if 'cosine' in names[i]:
        index.add(utils.normalize(features_db))
        D, I = index.search(utils.normalize(features_query), k)
    
    else:
        index.add(features_db)
        D, I = index.search(features_query, k)
        
    times.append(time.time() - t0)

    recall, _ = utils.recall(I, mapping_db, mapping_query)
    recalls.append(recall)
    
    # Free memory while keeping list length consant
    indices[i] = 0
    del index


plt.figure()
for i in range(len(names)):
    if 'JS' in names[i]:
        plt.scatter(recalls[i], times[i], color='red', marker='*', s=100)
    elif 'cosine' in names[i]:
        plt.scatter(recalls[i], times[i], color='blue', marker='o', s=100)
    elif 'L2' in names[i]:
        plt.scatter(recalls[i], times[i], color='green', marker='x', s=100)
        
lims = plt.gca().get_xlim()
space = (lims[1] - lims[0])/40
for i in range(len(names)):
    plt.annotate(names[i].split(' ')[1], (recalls[i] + space, times[i] + space), fontsize=16)

plt.xlabel('Recall@1')
plt.ylabel('Search time [s]')
plt.legend(['J-S', 'cosine', 'L2'])
        
plt.savefig('test.pdf', bbox_inches='tight')
plt.show()
        

