#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:28:49 2022

@author: cyrilvallez
"""

import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from helpers import utils
from helpers import plot_config

algorithm = 'SimCLR v2 ResNet50 2x'
dataset1 = 'Kaggle_memes'
dataset2 = 'Kaggle_templates'

_, mapping = utils.combine_features(algorithm, dataset1, dataset2)

distances = np.load('distances_all_memes_L2.npy')

Z = linkage(distances, method='ward')

thresholds = np.linspace(10, 100, 20)

rng = np.random.default_rng(seed=112)

folder = 'Clustering/Contrastive_net/'
os.makedirs(folder, exist_ok=True)

for i, threshold in enumerate(thresholds):
    
    current_dir = folder + f'threshold_{threshold:.3f}/'
    os.makedirs(current_dir, exist_ok=True)
    
    clusters = fcluster(Z, threshold, criterion='distance')
    N_clusters = int(max(clusters))
    
    for cluster_idx in range(1, N_clusters+1):
        
        images = mapping[clusters == cluster_idx]
        if len(images) > 10:
            representatives = rng.choice(images, size=10, replace=False)
        else:
            representatives = images
            
        representation = utils.cluster_representation(representatives)
        representation.save(current_dir + f'{cluster_idx}.png')
        
   

"""
plt.figure(figsize=(10,10))
dendrogram(Z)
plt.xlabel('Image number')
plt.ylabel('Cluster distance')
plt.savefig('test.pdf', bbox_inches='tight')
plt.show()
"""
