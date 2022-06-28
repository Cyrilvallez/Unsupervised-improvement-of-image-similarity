#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:05:38 2022

@author: cyrilvallez
"""

import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class SimCLR_Transforms(object):
    """
    Represent the data augmentation policy of SimCLR v1 and v2.

    Parameters
    ----------
    size : tuple, optional
        Final size for resizing the images. The default is (224, 224).
    jitter : float, optional
        The color jitter strength. The default is 1..

    """
    
    def __init__(self, size=224, jitter=1.):
        
        self.size = size
        self.jitter = jitter

        # The following transformations are implementations of the transformations
        # described in the paper on SimCLR v1 (SimCLR v2 uses the same)
        transforms = []
        
        # Random cropping and resizing
        random_resize = T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC)
        transforms.append(random_resize)
        
        # Randomly applied flip
        random_flip = T.RandomHorizontalFlip(p=0.5)
        transforms.append(random_flip)
        
        # Randomly applied random color jitter
        color_jitter = T.ColorJitter(0.8*jitter, 0.8*jitter, 0.8*jitter,
                                     0.2*jitter)
        random_color_jitter = T.RandomApply([color_jitter], p=0.8)
        transforms.append(random_color_jitter)
        
        # Randomly applied grayscale
        random_gray = T.RandomGrayscale(p=0.2)
        transforms.append(random_gray)
        
        # Randomly applied random gaussian blur
        kernel_size = int(0.1*size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        gaussian = T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        random_gaussian = T.RandomApply([gaussian], p=0.5)
        transforms.append(random_gaussian)
        
        # Converts to Tensor
        transforms.append(T.ToTensor())
        
        self.transforms = T.Compose(transforms)
        
    
    def __call__(self, image):
        """
        Draw 2 data augmentations from an image.

        Parameters
        ----------
        image : PIL image
            The image from which to augment.

        Returns
        -------
        x1 : Tensor
            The first data augmentation.
        x2 : Tensor
            The second data augmentation.

        """
        
        x1 = self.transforms(image)
        x2 = self.transforms(image)
        
        return x1, x2
    
    
    
class ImageDataset(Dataset):
    """
    Class representing a dataset of images.
    
    Parameters
    ----------
    dataset_path : Str or list of str
        Path to find the images or list of such paths, representing the images.
    size : tuple, optional
        Final size for resizing the images. The default is (224, 224).
    jitter : float, optional
        The color jitter strength. The default is 1..

    """
    
    def __init__(self, dataset_path, size=224, jitter=1.):
        
        super().__init__()
        
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
        
        self.transforms = SimCLR_Transforms(size=size, jitter=jitter)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Decode the binary string to normal string
        image_path = self.images[index].decode()
        image = Image.open(image_path).convert('RGB')
        
        # Draw the 2 data augmentation from the augmentation policy
        x1, x2 = self.transforms(image)
            
        return x1, x2
        
    
