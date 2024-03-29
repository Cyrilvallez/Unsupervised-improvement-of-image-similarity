#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:41:22 2022

@author: cyrilvallez
"""

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageFile

# We set this to True because 4 images in the Flickr500K dataset are corrupted 
# (this does not come from the download --> this was tested with different browsers
# multiple times). They are numbers 59898, 104442, 107349 and 108460. One is almost 
# completely lost (most image is grey), two are about half lost, and one has little
# damage. We still keep them by loading the image parts we can.\

ImageFile.LOAD_TRUNCATED_IMAGES = True
# AK: those are configs and are best managed by loading cnf.XXX rather than hard-coded


class ImageDataset(Dataset):
    """
    Class representing a dataset of images.
    
    Parameters
    ----------
    dataset_path : Str or list of str
        Path to find the images or list of such paths, representing the images.
    name : str
        A name for the dataset, detailing the images.
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. The default is None.
    """
    
    def __init__(self, dataset_path, name, transforms=None):
        
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
        
        self.name = name
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Decode the binary string to normal string
        image_path = self.images[index].decode()
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            
        return (image, image_path)
    

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
                      if not folder.startswith('.')]  # AK: Pathlib is your friend again.
        
        imgs = []
        for folder in subfolders:
            imgs += [folder + file for file in os.listdir(folder) if not file.startswith('.')]
            
        # Sort the images according the their number in the name
        imgs.sort(key=lambda x: int(x.rsplit('/', 1)[1].split('.', 1)[0]))
        # AK: Pathlib is your friend, yet again
        
        # Conversion to numpy byte array is important when using different
        # workers in Dataloader to avoid memory problems (see issue 
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)
        self.images = np.array(imgs).astype(np.string_)
        self.name = 'Flickr500K'
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Decode the binary string to normal string
        image_path = self.images[index].decode()
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            
        return (image, image_path)
    

class ImageWithDistractorDataset(FlickrDataset):
    """
    Class representing a dataset of images in which we add the Flickr500K images
    as distractors.
    
    Parameters
    ----------
    dataset_path : Str or list of str
        Path to find the images or list of such paths, representing the images 
        completing the Flickr dataset.
    name : str
        A name for the dataset completing the 500K flickr images.
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. The default is None.
    """
    
    def __init__(self, dataset_path, name, transforms=None):
        
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
            
        self.name = name + '+500K'
            
            
def collate(batch):
    """
    Custom collate function to use with PyTorch dataloader

    Parameters
    ----------
    batch : List of tuples 
        Corresponds to a batch of examples from a dataset.

    Returns
    -------
    Tuple
        Tuple representing a full batch containing a tuple of PIL images and
        other tuples of names.

    """
    
    imgs, names = zip(*batch)
    return (imgs, names)

    
VALID_DATASET_NAMES = [
    'Kaggle_templates',
    'Kaggle_memes',
    'BSDS500_original',
    'BSDS500_attacks',
    'Flickr500K',
    'all_memes'
    ]

DATASET_DIMS = {
    'Kaggle_templates': 250,
    'Kaggle_memes': 43660,
    'BSDS500_original': 500,
    'BSDS500_attacks': 11600,
    'Flickr500K': 500000,
    'all_memes': 43910,
    }

    
def create_dataset(dataset_name, transforms):
    """
    Create a dataset from certain given string for easy access.

    Parameters
    ----------
    dataset_name : string
        Identifier of the dataset.
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. 

    Raises
    ------
    ValueError
        If `dataset_name` is invalid.

    Returns
    -------
    ImageDataset
        The desired dataset of images.

    """
    
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(f'The dataset name must be one of {*VALID_DATASET_NAMES,}.')
    
    if dataset_name == 'Kaggle_templates':
        path = 'Datasets/Kaggle_memes/Templates/'
    elif dataset_name == 'Kaggle_memes':
        path = 'Datasets/Kaggle_memes/Memes/'
    elif dataset_name == 'BSDS500_original':
        path = 'Datasets/BSDS500/Original/'
    elif dataset_name == 'BSDS500_attacks':
        path = 'Datasets/BSDS500/Attacks/'
    elif dataset_name == 'Flickr500K':
        return FlickrDataset(transforms)
    elif dataset_name == 'all_memes':
        return all_memes_dataset(transforms)
    
    return ImageDataset(path, dataset_name, transforms=transforms)



def all_memes_dataset(transforms):
    """
    Return a dataset containing all memes, templates and actual memes.

    Parameters
    ----------
    transforms : Torch transforms
        Transforms to apply to each image. If not specified, transforms are
        not applied, and raw PIL image will be returned. 

    Returns
    -------
    ImageDataset
        The dataset corresponding to the merge of the two other.

    """
    
    dataset1 = create_dataset('Kaggle_memes', transforms)
    dataset2 = create_dataset('Kaggle_templates', transforms)
    
    imgs = dataset1.images.tolist() + dataset2.images.tolist()
    name = 'all_memes'
    
    return ImageDataset(imgs, name, transforms)


