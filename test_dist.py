#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:05:50 2022

@author: cyrilvallez
"""

import os
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
    
def main(rank, args):
    
    setup(rank, args.world_size)
    
    print(args.test)
    
    cleanup()
    
    

def run_demo(function, args):
    
    mp.spawn(function, args=(args,), nprocs=args.world_size, join=True)
    

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='Test')
    
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--test', type=str, default='Hello dear')
    
    args = parser.parse_args()
    
    run_demo(main, args)
    
    
#%%
import argparse

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--world_size', type=int, default=2)
parser.add_argument('--test', type=str, default='Hello dear')

args = parser.parse_args()