#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:05:50 2022

@author: cyrilvallez
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
    
def main(rank, world_size):
    
    setup(rank, world_size)
    
    tensor_list = [torch.zeros(2, dtype=torch.int64).to('cuda') for _ in range(2)]
    
    tensor = torch.rand(2).to('cuda')
    print('Simple tensor :')
    print(tensor)
    
    dist.all_gather(tensor_list, tensor)
    print('Gathered output :')
    print(tensor_list)
    
    
    cleanup()

def run_demo(demo_fn, world_size):
    
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)
    
    
if __name__ == '__main__':
    
    run_demo(main, 2)