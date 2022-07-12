#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:28:09 2022

@author: cyrilvallez
"""

import torch
import torch.nn as nn

# TODO : change location or duplicate this file
from extractor.SimCLRv2 import resnet as SIMv2

class SimCLR(nn.Module):
    """
    Module representing SimCLR. 

    Parameters
    ----------
    encoder : torch.nn.Module
        Model representing the encoder.
    contrastive_head : torch.nn.Module
        Model representing the contrastive/projective head.
        
    """
    
    def __init__(self, encoder, contrastive_head):
        
        super().__init__()
        self.encoder = encoder
        self.contrastive_head = contrastive_head
        
    def forward(self, x1, x2):
        """
        Implements the forward.

        Parameters
        ----------
        x1 : Tensor
            The batch of first data augmentation.
        x2 : Tensor
            The batch of second data augmentation.

        Returns
        -------
        z1 : Tensor
            Output corresponding to 1st data augmentation.
        z2 : Tensor
            Output corresponding to 2nd data augmentation.

        """
    
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
    
        z1 = self.contrastive_head(h1)
        z2 = self.contrastive_head(h2)
        
        return z1, z2
        
        
    def save(self, path):
        """
        Save the model, separating the encoder and the head.

        Parameters
        ----------
        path : str
            Path where to save the model.

        Returns
        -------
        None.

        """
        
        state = {'encoder': self.encoder.state_dict(),
                 'head': self.contrastive_head.state_dict()}
        torch.save(state, path)
        
        
    @staticmethod
    def load(path=None, depth=50, width=2, sk_ratio=0.0625):
        """
        Easily load a SimCLR module.

        Parameters
        ----------
        path : str, optional
            Path to the saved model. Load the original by default.
            The default is None.
        depth : int, optional
            The depth of the resnet used as encoder. This is needed to 
            reconstruct the module. The default is 50.
        width : int, optional
            The width of the resnet used as encoder. This is needed to 
            reconstruct the module. The default is 2.
        sk_ratio : float, optional
            The sk_ratio of the resnet used as encoder. This is needed to 
            reconstruct the module. The default is 0.0625.

        Returns
        -------
        torch.nn.Module
            The simCLR module.

        """
        
        if path is None:
            sk = 'sk1' if sk_ratio == 0.0625 else 'sk0'
            path = f'extractor/SimCLRv2/Pretrained/r{depth}_{width}x_{sk}_ema.pth'
        encoder, head = SIMv2.get_resnet(depth=depth, width_multiplier=width,
                                        sk_ratio=sk_ratio)
        checkpoint = torch.load(path)
        try:
            encoder.load_state_dict(checkpoint['encoder'])
        except KeyError:
            encoder.load_state_dict(checkpoint['resnet'])
        head.load_state_dict(checkpoint['head'])
        
        return SimCLR(encoder, head)
    
    
    @staticmethod
    def load_encoder(path=None, depth=50, width=2, sk_ratio=0.0625):
        """
        Easily load the encoder of a SimCLR module.

        Parameters
        ----------
        path : str, optional
            Path to the saved model. Load the original by default.
            The default is None.
        depth : int, optional
            The depth of the resnet used as encoder. This is needed to 
            reconstruct the module. The default is 50.
        width : int, optional
            The width of the resnet used as encoder. This is needed to 
            reconstruct the module. The default is 2.
        sk_ratio : float, optional
            The sk_ratio of the resnet used as encoder. This is needed to 
            reconstruct the module. The default is 0.0625.

        Returns
        -------
        torch.nn.Module
            The encoder module.

        """
        
        if path is None:
            sk = 'sk1' if sk_ratio == 0.0625 else 'sk0'
            path = f'extractor/SimCLRv2/Pretrained/r{depth}_{width}x_{sk}_ema.pth'
        encoder, _ = SIMv2.get_resnet(depth=depth, width_multiplier=width,
                                        sk_ratio=sk_ratio)
        checkpoint = torch.load(path)
        try:
            encoder.load_state_dict(checkpoint['encoder'])
        except KeyError:
            encoder.load_state_dict(checkpoint['resnet'])
        
        return encoder
        
    