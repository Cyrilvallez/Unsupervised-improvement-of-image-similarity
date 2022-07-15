#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:41:19 2022

@author: cyrilvallez
"""

import torch.multiprocessing as mp

from finetuning.training import main, parse_args


if __name__ == '__main__':
    
    args = parse_args()
    
    if args.gpus > 1:
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
        
    else:
        main(0, args)
    
    
    