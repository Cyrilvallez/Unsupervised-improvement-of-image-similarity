#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:54:08 2022

@author: cyrilvallez
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from PIL import Image
from tqdm import tqdm

from Extract_features.SimCLRv1 import resnet_wider as SIMv1
from Extract_features.SimCLRv2 import resnet as SIMv2

path = os.path.abspath(__file__)
current_folder = os.path.dirname(path)

class ImageDataset(Dataset):
    """
    Class representing a dataset of images.
    """
    
    def __init__(self, path_to_imgs, transforms):
        
        if (type(path_to_imgs) == str or type(path_to_imgs) == np.str_):
            # Take files in folder without hidden files (e.g .DS_Store)
            imgs = [file for file in os.listdir(path_to_imgs) if not file.startswith('.')]
            imgs.sort(key=lambda x: int(x.split('.', 1)[0]))
            self.img_paths = [path_to_imgs + name for name in imgs]
        elif type(path_to_imgs) == list:
            self.img_paths = path_to_imgs
        
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        image = self.transforms(image)
        
        try:
            name = self.img_paths[index].rsplit('/', 1)[1]
        except IndexError:
            name = self.img_paths[index]
            
        # Removes the extension (name.jpg -> name)
        name = name.rsplit('.', 1)[0]
            
        return (image, name)
    
    
def load_inception_v3(device='cuda'):
    """
    Load the inception net v3 from Pytorch.

    Parameters
    ----------
    device : str, optional
        Device on which to load the model. The default is 'cuda'.

    Returns
    -------
    inception : PyTorch model
        Inception net v3.

    """
    
    # Load the model 
    inception = models.inception_v3(pretrained=True, transform_input=False)
    # Overrides last Linear layer
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(torch.device(device))
    
    return inception


def load_resnet(depth, width):
    """
    Load a resnet model with given depth and width.

    Parameters
    ----------
    depth : int
        Depth of the ResNet model
    width : int
        Width multiplier.

    Returns
    -------
    load : function
        A loader function for the ResNet model.

    """
    
    if depth==50 and width==1:
        loader = models.resnet50
    elif depth==101 and width==1:
        loader = models.resnet101
    elif depth==152 and width==1:
        loader = models.resnet152
    elif depth==50 and width==2:
        loader = models.wide_resnet50_2
    elif depth==101 and width==2:
        loader = models.wide_resnet101_2
    else:
        raise ValueError('This combination of depth and width is not valid.')
    
    def load(device='cuda'):
        
        # Load the model 
        resnet = loader(pretrained=True)
        # Overrides last Linear layer
        resnet.fc = nn.Identity()
        resnet.eval()
        resnet.to(torch.device(device))
    
        return resnet
    
    return load


def load_efficientnet_b7(device='cuda'):
    """
    Load the efficient net b7 from Pytorch.

    Parameters
    ----------
    device : str, optional
        Device on which to load the model. The default is 'cuda'.

    Returns
    -------
    efficientnet : PyTorch model
        Efficient net b7.

    """
    
    # Load the model 
    efficientnet = models.efficientnet_b7(pretrained=True)
    # Overrides last Linear layer and adds a relu instead 
    efficientnet.classifier = nn.ReLU()
    efficientnet.eval()
    efficientnet.to(torch.device(device))
    
    return efficientnet


def load_simclr_v1(width):
    """
    Load the simclr v1 ResNet50 model with the given width.

    Parameters
    ----------
    width : int
        Width multiplier.

    Returns
    -------
    load : function
        A loader function for the simclr v1 ResNet50 1x model.

    """
    
    checkpoint_file = current_folder + f'/SimCLRv1/Pretrained/resnet50-{width}x.pth'
    
    def load(device='cuda'):
        
        # Load the model 
        simclr = SIMv1.get_resnet(width=width)
        checkpoint = torch.load(checkpoint_file)
        simclr.load_state_dict(checkpoint['state_dict'])
        simclr.fc = nn.Identity()
        simclr.eval()
        simclr.to(torch.device(device))
    
        return simclr
    
    return load



def load_simclr_v2(depth, width, selective_kernel=True):
    """
    Load the simclr v2 ResNet model with given depth and width.

    Parameters
    ----------
    depth : int
        Depth of the ResNet model
    width : int
        Width multiplier.
    selective_kernel : Boolean
        Whether to use a selective kernel.

    Returns
    -------
    load : function
        a Loader function for the simclr v2 ResNet model with given depth and width.

    """
    
    if selective_kernel:
        checkpoint_file = current_folder + f'/SimCLRv2/Pretrained/r{depth}_{width}x_sk1_ema.pth'
    else:
        checkpoint_file = current_folder + f'/SimCLRv2/Pretrained/r{depth}_{width}x_ema.pth'
    
    def load(device='cuda'):
        
        # Load the model 
        sk_ratio = 0.0625 if selective_kernel else 0
        simclr, _ = SIMv2.get_resnet(depth=depth, width_multiplier=width, sk_ratio=sk_ratio)
        checkpoint = torch.load(checkpoint_file)
        simclr.load_state_dict(checkpoint['resnet'])
        simclr.eval()
        simclr.to(torch.device(device))
    
        return simclr
    
    
    return load


# Mapping from model name to actual models
MODEL_LOADER = {
    'Inception v3': load_inception_v3,
    'ResNet50 1x': load_resnet(50, 1),
    'ResNet101 1x': load_resnet(101, 1),
    'ResNet152 1x': load_resnet(152, 1),
    'ResNet50 2x': load_resnet(50, 2),
    'ResNet101 2x': load_resnet(101, 2),
    'EfficientNet B7': load_efficientnet_b7,
    'SimCLR v1 ResNet50 1x': load_simclr_v1(width=1),
    'SimCLR v1 ResNet50 2x': load_simclr_v1(width=2),
    'SimCLR v1 ResNet50 4x': load_simclr_v1(width=4),
    'SimCLR v2 ResNet50 2x': load_simclr_v2(depth=50, width=2, selective_kernel=True),
    'SimCLR v2 ResNet101 2x': load_simclr_v2(depth=101, width=2, selective_kernel=True),
    'SimCLR v2 ResNet152 3x': load_simclr_v2(depth=152, width=3, selective_kernel=True),
    }


# Transforms for all SimCLR models
SIMCLR_TRANSFORMS = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor()
    ])


# Pretrained pytorch models transforms
RESNET_TRANSFORMS = T.Compose([
    T.Resize((256,256), interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Mapping from model name to pre-processing transforms
MODEL_TRANSFORMS = {
    'Inception v3': T.Compose([
        T.Resize((299,299), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    
    'ResNet50 1x' : RESNET_TRANSFORMS,
    
    'ResNet101 1x' : RESNET_TRANSFORMS,
    
    'ResNet152 1x' : RESNET_TRANSFORMS,
    
    'ResNet50 2x': RESNET_TRANSFORMS,
    
    'ResNet101 2x': RESNET_TRANSFORMS,
    
    'EfficientNet B7': T.Compose([
        T.Resize((600,600), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    
    'SimCLR v1 ResNet50 1x': SIMCLR_TRANSFORMS,
    
    'SimCLR v1 ResNet50 2x': SIMCLR_TRANSFORMS,
    
    'SimCLR v1 ResNet50 4x': SIMCLR_TRANSFORMS,
    
    'SimCLR v2 ResNet50 2x': SIMCLR_TRANSFORMS,
    
    'SimCLR v2 ResNet101 2x': SIMCLR_TRANSFORMS,
    
    'SimCLR v2 ResNet152 3x': SIMCLR_TRANSFORMS,
    
    }

    

def extract_and_save_features(model, dataset, filename, batch_size=256):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    file = open(filename, 'w+')
    
    for images, names in tqdm(dataloader):
        
        features = model(images).numpy()
        M = len(features)
        N = len(features[0])
        
        for i in range(M):
            
            file.write(f'{names[i]} ')
            
            for j in range(N):
                
                file.write(f'{features[i,j]:.18e} ')
                
            file.write(' \n')
            
    file.close()
    
    
    


if __name__ == "__main__":
    
    model_name = 'SimCLR v2 ResNet50 2x'
    dataset_path = 'Datasets/'
    
    model = MODEL_LOADER[model_name](device='cuda')
    transforms = MODEL_TRANSFORMS[model_name]
    
    dataset = ImageDataset(dataset_path, transforms)
    
    filename = 'test.txt'
    extract_and_save_features(model, dataset, filename=filename, batch_size=512)