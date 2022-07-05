#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:58:28 2022

@author: cyrilvallez
"""

import extractor
from clustering import tools
from finetuning.simclr import SimCLR

transforms = extractor.SIMCLR_TRANSFORMS
dataset = extractor.all_memes_dataset(transforms)

path = 'first_test_models/2022-06-30_20:03:28/epoch_10.pth'
model = SimCLR.load_encoder(path)

features, mapping = extractor.extract_neural(model, dataset, batch_size=256)

features2, mapping2 = tools.get_dataset('full_dataset', 'SimCLR v2 ResNet50 2x')

assert mapping == mapping2