#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

import extractor
from tqdm import tqdm

# method_name = 'SimCLR v2 ResNet50 2x'
method_name = 'Dhash'
hash_size = 8
batch_size = 256


datasets = extractor.VALID_DATASET_NAMES

# Perceptual
if 'hash' in method_name:
    for dataset in tqdm(datasets):
        dataset = extractor.create_dataset(dataset, None)
        extractor.extract_and_save_perceptual(method_name, dataset, hash_size=hash_size,
                                              workers=3)
       
# Neural
else:
    for dataset in tqdm(datasets):
        transforms = extractor.MODEL_TRANSFORMS[method_name]
        dataset = extractor.create_dataset(dataset, transforms)
        extractor.extract_and_save_neural(method_name, dataset, batch_size=batch_size, workers=3)

