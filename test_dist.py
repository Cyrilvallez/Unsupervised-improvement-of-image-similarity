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
from torch.autograd import gradcheck
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import math
from finetuning import loss

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    
def get_model():
    torch.manual_seed(20)
    return ToyModel()
    
def get_batch(rank, part=True):
    torch.manual_seed(20)
    batch1 = torch.rand(40, 10)
    batch2 = torch.rand(40, 10)
    
    if part:
        return batch1[20*rank:20*(rank+1)], batch2[20*rank:20*(rank+1)]
    else:
        return batch1, batch2
        

# =============================================================================
# 
# =============================================================================
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
    
def main(rank, world_size):
    
    setup(rank, world_size)
    
    model = DDP(get_model().cuda(rank), device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    batch1, batch2 = get_batch(rank)
    batch1, batch2 = batch1.cuda(rank), batch2.cuda(rank)
    output1 = model(batch1)
    output2 = model(batch2)
    criterion = loss.NT_Xent2(batch1.shape[0], 3., world_size)
    L = criterion(output1, output2)
    print(f'loss : {L}')
    L.backward()
    optimizer.step()
    
    print(f'Parameters : {list(model.parameters())}')
    
    """
    x = torch.tensor([42.], requires_grad=True).cuda(rank)
    x.retain_grad()
    xs = torch.stack(func(x))
    # xs = func(x)

    xs *= rank # multiply by rank
    xs.sum().backward()

    print(f"rank: {rank}, x:{x.grad}")

    # gradient should be equal to the rank for each process
    # so total gradient should be the sum of arange
    assert torch.allclose(
        x.grad,
        torch.arange(world_size).sum().float()
    )
    """

    """
    tensor = (torch.rand(2,2) + 2*rank).cuda(rank) 
    print('Simple tensor :')
    print(tensor)
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(tensor_list, tensor)
    print('Gathered output :')
    print(tensor_list)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    print('Reducted output :')
    print(tensor)
    """
    if rank == 0:
        torch.save(model.state_dict(), 'test.checkpoint')
    cleanup()
    
    

def run_demo(function, world_size):
    
    mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)
    

if __name__ == '__main__':
    
    run_demo(main, 2)
    
    print('Without DDP :')
    model = get_model().cuda(0)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    batch1, batch2 = get_batch(0, part=False)
    batch1, batch2 = batch1.cuda(0), batch2.cuda(0)
    output1 = model(batch1)
    output2 = model(batch2)
    criterion = loss.NT_Xent2(batch1.shape[0], 3., world_size)
    L = criterion(output1, output2)
    print(f'loss : {L}')
    L.backward()
    optimizer.step()
    print(f'Parameters : {list(model.parameters())}')
    
    previous = get_model().cuda(0)
    state = torch.load('test.checkpoint')
    state = {a.split('.', 1)[1]:b for a,b in state.items()}
    previous.load_state_dict(state)
    
    print(f'ALL SAME : {max([torch.max(torch.abs((a - b))) for a,b in zip(list(model.parameters()), list(previous.parameters()))])}')
    
    