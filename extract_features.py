#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

import extractor

# method_name = 'SimCLR v2 ResNet50 2x'
method_name = 'Dhash'
hash_size = 8
batch_size = 1024


datasets = extractor.VALID_DATASET_NAMES

# Perceptual
if 'hash' in method_name:
    for dataset in datasets:
        dataset = extractor.create_dataset(dataset, None)
        extractor.extract_and_save_perceptual(method_name, dataset, hash_size=hash_size)
       
# Neural
else:
    for dataset in datasets:
        transforms = extractor.MODEL_TRANSFORMS[method_name]
        dataset = extractor.create_dataset(dataset, transforms)
        extractor.extract_and_save_neural(method_name, dataset, batch_size=batch_size, workers=8)

