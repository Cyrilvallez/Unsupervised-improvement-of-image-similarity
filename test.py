#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import shutil
from PIL import Image
from helpers import utils
import torch
import faiss

algorithm = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_memes'

features, _ = utils.load_features(algorithm, dataset)
# features = torch.tensor(features).to('cuda')

# res = torch.cdist(features, features).cpu().numpy()
# np.save('distances.npy', res)

#%%

index = faiss.IndexFlatIP(features.shape[1])
index = faiss.index_cpu_to_all_gpus(index)

index.add(features)
D, I = index.search(features, 10)

res = np.zeros((features.shape[0], features.shape[0]), dtype='float32')

for i in range(len(I)):
    res[i, I[i,:]] = D[i, :]
    
np.save('distances_neighbors.npy', res)


