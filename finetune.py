#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:41:19 2022

@author: cyrilvallez
"""

from finetuning.training import main, parse_args
  

if __name__ == '__main__':
    
    """
    model = SimCLR.load()
    epochs = 20
    lr = 1e-3
    optimizer = 'lars'
    decay = 0
    momentum = 0
    nesterov = False
    
    optimizer, scheduler = get_optimizer(model, epochs, optimizer, lr, decay,
                                         momentum, nesterov)
    
    for i in range(epochs):
        print(scheduler.get_last_lr()[0])
        scheduler.step()
    """
        
    args = parse_args()
    
    
    
    
    