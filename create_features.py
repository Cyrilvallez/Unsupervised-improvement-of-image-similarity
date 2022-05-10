#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

from Extract_features import neural_features as nf
from Extract_features import datasets

model_name = 'SimCLR v2 ResNet50 2x'

model = nf.MODEL_LOADER[model_name](device='cuda')
transforms = nf.MODEL_TRANSFORMS[model_name]

dataset = datasets.FlickrDataset(transforms=transforms)

filename = 'Flickr500K-SimCLR_v2_ResNet50_2x'

nf.extract_and_save_features(model, dataset, filename, batch_size=1024, workers=8)


