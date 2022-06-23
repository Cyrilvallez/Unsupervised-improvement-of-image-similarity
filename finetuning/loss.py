#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:20:16 2022

@author: cyrilvallez
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class Gather(torch.autograd.Function):
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
    
    
    
def gather(tensor):
    """
    Perform an `all_gather` operation to gather tensors from all GPUs, while
    still enabling backward flow of the gradients. This is needed since the 
    loss must be computed between all examples of a mini-batch, and not only
    between examples on each GPU separately.

    Parameters
    ----------
    tensor : Tensor
        The batch from one process.

    Returns
    -------
    Tensor
        Batch corresponding to the concatenation of batches across all processes.

    """
    
    return Gather.apply(tensor)
    
    
    
class NT_Xent(nn.Module):
    
    def __init__(self, process_batch_size, temperature):
        
        super(NT_Xent, self).__init__()
        self.process_batch_size = process_batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size, world_size)
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

    def forward(self, z1, z2):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        if dist.is_available() and dist.is_initialized():
            N = 2*self.process_batch_size*dist.get_world_size()
            z1_tot = gather(z1)
            z2_tot = gather(z2)
        else:
            N = 2*self.process_batch_size
            z1_tot = z1
            z2_tot = z2

        z_tot = torch.cat((z1_tot, z2_tot), dim=0)
        z = torch.cat((z1, z2), dim=0)
        

        

    