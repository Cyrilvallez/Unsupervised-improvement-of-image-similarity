#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:41:22 2022

@author: cyrilvallez
"""

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

class ImageDataset(Dataset):
    """
    Class representing a dataset of images.
    
    Parameters
    ----------
    dataset_path : Str or list of str
        Path to find the images or list of such paths, representing the images.
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. The default is None.
    """
    
    def __init__(self, dataset_path, transforms=None):
        
        if (type(dataset_path) == str or type(dataset_path) == np.str_):
            # Append last `/` if not present
            if dataset_path[-1] != '/':
                dataset_path += '/'
            # Take files in folder without hidden files (e.g .DS_Store)
            self.images = [dataset_path + file for file in os.listdir(dataset_path) \
                          if not file.startswith('.')]
            
        elif type(dataset_path) == list:
            self.images = dataset_path
        
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        
        try:
            name = self.images[index].rsplit('/', 1)[1]
        except IndexError:
            name = self.images[index]
            
        # Removes the extension (name.jpg -> name)
        name = name.rsplit('.', 1)[0]
            
        return (image, name)
    

class FlickrDataset(Dataset):
    """
    Class representing the Flickr500K dataset of images.
    
    Parameters
    ----------
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. The default is None.
    """
    
    def __init__(self, transforms=None):
        
        dataset_path = 'Datasets/Flickr500K/'
        # Take subfolders without hidden files (e.g .DS_Store)
        subfolders = [dataset_path + folder + '/' for folder in os.listdir(dataset_path) \
                      if not folder.startswith('.')]
        
        imgs = []
        for folder in subfolders:
            imgs += [folder + file for file in os.listdir(folder) if not file.startswith('.')]
            
        # Sort the images according the their number in the name
        imgs.sort(key=lambda x: int(x.rsplit('/', 1)[1].split('.', 1)[0]))
        
        self.images = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        
        try:
            name = self.images[index].rsplit('/', 1)[1]
        except IndexError:
            name = self.images[index]
            
        # Removes the extension (name.jpg -> name)
        name = name.rsplit('.', 1)[0]
            
        return (image, name)
    

class ImageWithDistractorDataset(FlickrDataset):
    """
    Class representing a dataset of images in which we add the Flickr500K images
    as distractors.
    
    Parameters
    ----------
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. The default is None.
    """
    
    def __init__(self, dataset_path, transforms=None):
        
        super().__init__(transforms)
        
        if (type(dataset_path) == str or type(dataset_path) == np.str_):
            # Append last `/` if not present
            if dataset_path[-1] != '/':
                dataset_path += '/'
            # Take files in folder without hidden files (e.g .DS_Store)
            self.images += [dataset_path + file for file in os.listdir(dataset_path) \
                          if not file.startswith('.')]
            
        elif type(dataset_path) == list:
            self.images += dataset_path
        