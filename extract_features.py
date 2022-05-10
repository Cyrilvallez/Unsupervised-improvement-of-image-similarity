#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

import extractor

"""
# Neural
model_name = 'SimCLR v2 ResNet50 2x'
transforms = extractor.MODEL_TRANSFORMS[model_name]
dataset = extractor.FlickrDataset(transforms=transforms)
filename = 'Features/Flickr500K-SimCLR_v2_ResNet50_2x'
extractor.extract_and_save_neural(model_name, dataset, filename, batch_size=1024, workers=8)
"""

# Perceptual
algo_name = 'Dhash'
dataset = extractor.FlickrDataset()
filename = 'Features/Flickr500K-Dhash'
extractor.extract_and_save_perceptual(algo_name, dataset, filename, hash_size=8)