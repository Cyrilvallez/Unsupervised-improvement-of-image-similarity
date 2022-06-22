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
    
    tensor = (torch.rand(2) + 2*rank).cuda(rank) 
    print('Simple tensor :')
    print(tensor)
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(tensor_list, tensor)
    print('Gathered output :')
    print(tensor_list)
    reduction = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    print('Reducted output :')
    print(reduction)
    
    cleanup()

def run_demo(function, world_size):
    
    mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)
    
    
if __name__ == '__main__':
    
    run_demo(main, 2)