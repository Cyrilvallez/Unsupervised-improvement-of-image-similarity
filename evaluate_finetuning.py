#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:58:28 2022

@author: cyrilvallez
"""

import numpy as np
import os

import extractor
from finetuning.simclr import SimCLR
from clustering import metrics
from clustering import tools



transforms = extractor.SIMCLR_TRANSFORMS
dataset = extractor.create_dataset('all_memes', transforms)

epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
paths = [f'first_test_models/2022-06-30_20:03:28/epoch_{a}.pth' for a in epochs]
    
all_diameters = []
all_centroids = []
    
for epoch, path in zip(epochs, paths):
        
    # Load model
    model = SimCLR.load_encoder(path)
    # Give the model a name
    name = f'SimCLR_finetuned_epoch_{epoch}'
    # Save features
    extractor.extract_and_save_neural(model, dataset, name=name)
    # Load the features of only the perceptually similar memes
    features, mapping = tools.get_features('clean_dataset', name)
    
    assignments = tools.compute_assignment_groundtruth(mapping)

    diameters = metrics.cluster_diameters(features, assignments, quantile=1.)
    all_diameters.append(diameters)
        
    centroids = metrics.cluster_centroids(features, assignments)
    all_centroids.append(centroids)
        
    
       
os.makedirs('Finetuning_eval', exist_ok=True)

for diameters, epoch in zip(all_diameters, epochs):
    np.save(f'Finetuning_eval/diameters_epochs_{epoch}.npy', diameters)
        
for centroids, epoch in zip(all_centroids, epochs):
    np.save(f'Finetuning_eval/centroids_epochs_{epoch}.npy', centroids)
        
 


