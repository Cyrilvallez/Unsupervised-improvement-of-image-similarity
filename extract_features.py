#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

import extractor


# Neural
model_name = 'SimCLR v2 ResNet50 2x'
transforms = extractor.MODEL_TRANSFORMS[model_name]
dataset = extractor.create_dataset('BSDS500_attacks', transforms)
extractor.extract_and_save_neural(model_name, dataset, batch_size=1024, workers=8)


"""
# Perceptual
algo_name = 'Dhash'
dataset = extractor.FlickrDataset()

extractor.extract_and_save_perceptual(algo_name, dataset, hash_size=8)
"""