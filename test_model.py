#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:11:22 2022

@author: cyrilvallez
"""
import torch
import torch.nn as nn

from finetuning.simclr import SimCLR
from extractor.SimCLRv1 import resnet_wider
from extractor.SimCLRv2 import resnet

model1 = SimCLR.load()
model2 = SimCLR.load('/Users/cyrilvallez/Desktop/Project2/test1_models/2022-07-12_09:52:24/epoch_2.pth', map_location=torch.device('cpu'))


"""
model1 = resnet_wider.get_resnet(width=4)
model2 = resnet_wider.get_resnet(width=4)

checkpoint1 = torch.load('/Users/cyrilvallez/Desktop/SimCLRv1/torch_checkpoints/Pretrained/resnet50-4x.pth')
model1.load_state_dict(checkpoint1['state_dict'])

checkpoint2 = torch.load('/Users/cyrilvallez/Desktop/Project2/extractor/SimCLRv1/Pretrained/resnet50-4x.pth')
model2.load_state_dict(checkpoint2['state_dict'])
"""
# model1.fc = nn.Identity()
# model2.fc = nn.Identity()

"""
model1, _ = resnet.get_resnet(depth=50, width_multiplier=2, sk_ratio=0.0625)
model2, _ = resnet.get_resnet(depth=50, width_multiplier=2, sk_ratio=0.0625)

checkpoint1 = torch.load('/Users/cyrilvallez/Desktop/SimCLRv2/torch_checkpoints/Pretrained/r50_2x_sk1_ema.pth')
model1.load_state_dict(checkpoint1['encoder'])

checkpoint2 = torch.load('/Users/cyrilvallez/Desktop/Project2/extractor/SimCLRv2/Pretrained/r50_2x_sk1_ema.pth')
model2.load_state_dict(checkpoint2['resnet'])
"""
model1.eval()
model2.eval()

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        
        
compare_models(model1, model2)
        
        