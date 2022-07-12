#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:11:22 2022

@author: cyrilvallez
"""
import torch

from finetuning.simclr import SimCLR

model1 = SimCLR.load()
model2 = SimCLR.load('/Users/cyrilvallez/Desktop/r50_2x_sk1_ema.pth')

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
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        
        
compare_models(model1, model2)
        
        