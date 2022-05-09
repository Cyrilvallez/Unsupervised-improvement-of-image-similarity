#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:44:42 2022

@author: cyrilvallez
"""

from Extract_features import neural_features as nf
from Extract_features import datasets

from tqdm import tqdm
import psutil

model_name = 'SimCLR v2 ResNet50 2x'

model = nf.MODEL_LOADER[model_name](device='cuda')
transforms = nf.MODEL_TRANSFORMS[model_name]

dataset = datasets.FlickrDataset(transforms=transforms)

nf.extract_features(model, dataset, batch_size=1024)

#%%
"""
from Extract_features import neural_features as nf
from Extract_features import datasets
from torch.utils.data import DataLoader

import psutil
from tqdm import tqdm

model_name = 'SimCLR v2 ResNet50 2x'
transforms = nf.MODEL_TRANSFORMS[model_name]

dataset = datasets.FlickrDataset(transforms=transforms)

dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=True,
                        num_workers=4, pin_memory=True)

print(f'Before : ram {psutil.virtual_memory().used/1e9:.2f} Gb')

i = 0
for images, names in tqdm(dataloader):
    i += 1
    print(f'Iter {i} ram : {psutil.virtual_memory().used/1e9:.2f} Gb')
"""     
    


