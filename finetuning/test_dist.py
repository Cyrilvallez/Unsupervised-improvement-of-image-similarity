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
        

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.mask = self.mask_correlated_samples(batch_size, self.world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(Gather2.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    
    
class Gather1(torch.autograd.Function):
    """
    Perform an `all_gather` operation to gather tensors from all GPUs, while
    still enabling backward flow of the gradients. This is needed since the 
    loss must be computed between all examples of a mini-batch, and not only
    between examples on each GPU separately.
    """
    
    @staticmethod
    def forward(ctx, tensor):
        
        # Save batch size to the context, for later use in backward
        ctx.batch_size = tensor.shape[0]
        
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        
        # The output is the complete batch
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        
        # grad_output must NEVER be modified inplace, thus we clone it
        grad_input = grad_output.clone()
        
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)

        # Get the indices of the batch of the current process
        idx_from = dist.get_rank()*ctx.batch_size
        idx_to = (dist.get_rank() + 1)*ctx.batch_size
        
        return grad_input[idx_from:idx_to]
    
    
class Gather2(torch.autograd.Function):
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
    
    
    
def loss1(out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = Gather1.apply(out_1)
            out_2_dist = Gather1.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
    
    
def loss2(z1, z2, temperature):
    criterion = NT_Xent(z1.shape[0], temperature)
    return criterion(z1, z2)
    

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
    L = loss2(output1, output2, 3.)
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
    L = loss2(output1, output2, 3.)
    print(f'loss : {L}')
    L.backward()
    optimizer.step()
    print(f'Parameters : {list(model.parameters())}')
    
    previous = get_model().cuda(0)
    state = torch.load('test.checkpoint')
    state = {a.split('.', 1)[1]:b for a,b in state.items()}
    previous.load_state_dict(state)
    
    print(f'ALL SAME : {max([torch.max(torch.abs((a - b))) for a,b in zip(list(model.parameters()), list(previous.parameters()))])}')
    
    