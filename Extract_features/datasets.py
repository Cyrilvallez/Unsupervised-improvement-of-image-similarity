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
            images = [dataset_path + file for file in os.listdir(dataset_path) \
                      if not file.startswith('.')]
                
            # Conversion to numpy byte array is important when using different
            # workers in Dataloader to avoid memory problems (see issue 
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)
            self.images = np.array(images).astype(np.string_)
            
        elif type(dataset_path) == list:
            self.images = np.array(dataset_path).astype(np.string_)
        
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Decode the binary string to normal string
        image_path = self.images[index].decode()
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        
        try:
            name = image_path.rsplit('/', 1)[1]
        except IndexError:
            name = image_path
            
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
        
        # Conversion to numpy byte array is important when using different
        # workers in Dataloader to avoid memory problems (see issue 
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)
        self.images = np.array(imgs).astype(np.string_)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Decode the binary string to normal string
        image_path = self.images[index].decode()
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        
        try:
            name = image_path.rsplit('/', 1)[1]
        except IndexError:
            name = image_path
            
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
            new_images = [dataset_path + file for file in os.listdir(dataset_path) \
                          if not file.startswith('.')]
                
            # Conversion to numpy byte array is important when using different
            # workers in Dataloader to avoid memory problems (see issue 
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)
            new_images = np.array(new_images).astype(np.string_)
            self.images = np.concatenate((self.images, new_images))
            
        elif type(dataset_path) == list:
            dataset_path = np.array(dataset_path).astype(np.string_)
            self.images = np.concatenate((self.images, dataset_path))
        
        
        
        
        
        
        
        
class FlickrDatasetBAD(Dataset):
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
        # Decode the binary string to normal string
        image_path = self.images[index]
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError:
            print(f'Bad index {index}', flush=True)
        if self.transforms is not None:
            image = self.transforms(image)
        
        try:
            name = image_path.rsplit('/', 1)[1]
        except IndexError:
            name = image_path
            
        # Removes the extension (name.jpg -> name)
        name = name.rsplit('.', 1)[0]
            
        return (image, name)