#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

import extractor

model_name = 'SimCLR v2 ResNet50 2x'
transforms = extractor.MODEL_TRANSFORMS[model_name]
dataset = extractor.FlickrDataset(transforms=transforms)
filename = 'Flickr500K-SimCLR_v2_ResNet50_2x'
extractor.extract_and_save_neural(model_name, dataset, filename, batch_size=1024, workers=8)


