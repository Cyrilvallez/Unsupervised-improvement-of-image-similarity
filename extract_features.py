#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

import extractor
from finetuning.simclr import SimCLR

# model = 'SimCLR v2 ResNet50 2x'
# model = 'Dhash'
model = SimCLR.load_encoder('first_test_models/2022-06-30_20:03:28/epoch_100.pth')
name = 'SimCLR finetuned'
transforms = extractor.SIMCLR_TRANSFORMS

hash_size = 8
batch_size = 256
workers = 6


datasets = extractor.VALID_DATASET_NAMES


if type(model) == str:
    
    # Perceptual
    if 'hash' in model:
        for dataset in datasets:
            dataset = extractor.create_dataset(dataset, None)
            extractor.extract_and_save_perceptual(model, dataset, hash_size=hash_size,
                                                  batch_size=batch_size, workers=workers)
       
    # Neural
    else:
        for dataset in datasets:
            transforms = extractor.MODEL_TRANSFORMS[model]
            dataset = extractor.create_dataset(dataset, transforms)
            extractor.extract_and_save_neural(model, dataset, batch_size=batch_size,
                                              workers=workers)
            
else:
    
    # Custom model neural
    for dataset in datasets:
        dataset = extractor.create_dataset(dataset, transforms)
        extractor.extract_and_save_neural(model, dataset, name=name, batch_size=batch_size,
                                          workers=workers)
    
