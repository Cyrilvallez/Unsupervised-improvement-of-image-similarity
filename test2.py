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

nf.extract_features(model, dataset, batch_size=1024)

#%%

# from Extract_features import neural_features as nf
# from Extract_features import datasets
# from torch.utils.data import DataLoader

# if __name__ == '__main__':

    # model_name = 'SimCLR v2 ResNet50 2x'
    # transforms = nf.MODEL_TRANSFORMS[model_name]

    # dataset = datasets.ImageDataset('Datasets/Flickr500K/0', transforms=transforms)

    # dataloader = DataLoader(dataset, batch_size=1028, shuffle=False, drop_last=True,
                        # num_workers=2)

    # for images, names in dataloader:
        # pass


