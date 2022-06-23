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
    
    
def loss(batch):
    return torch.nn.functional.pdist(batch).sum()
    
def get_batch(rank, part=True):
    torch.manual_seed(20)
    batch = torch.rand(40, 10)
    
    if part:
        return batch[20*rank:20*(rank+1)]
    else:
        return batch
        


class test1(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
class test2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
    
class test3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = torch.stack(grad_outs)
        dist.all_reduce(grad_outs)
        return grad_outs[dist.get_rank()]
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
    
def main(rank, world_size):
    
    setup(rank, world_size)
    
    func = test3.apply
    
    model = DDP(get_model().cuda(rank), device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    batch = get_batch(rank).cuda(rank)
    output = model(batch)
    full = func(output)
    if func == test1.apply:
        full = torch.cat(full, 0)
    L = loss(full)
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
    cleanup()
    
    

def run_demo(function, world_size):
    
    mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)
    
 
if __name__ == '__main__':
    
    run_demo(main, 2)
    
    print('Without DDP :')
    model = get_model().cuda(0)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    batch = get_batch(0, part=False).cuda(0)
    output = model(batch)
    L = loss(output)
    print(f'loss : {L}')
    L.backward()
    optimizer.step()
    print(f'Parameters : {list(model.parameters())}')
    
    