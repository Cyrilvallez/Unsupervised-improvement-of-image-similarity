#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

from Extract_features import neural_features as nf

model_name = 'SimCLR v2 ResNet50 2x'
dataset_path = 'Datasets/'

model = nf.MODEL_LOADER[model_name](device='cuda')
transforms = nf.MODEL_TRANSFORMS[model_name]

dataset = nf.ImageDataset(dataset_path, transforms)

filename = 'test.txt'
nf.extract_and_save_features(model, dataset, filename=filename, batch_size=512)