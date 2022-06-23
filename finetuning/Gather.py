#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:20:16 2022

@author: cyrilvallez
"""

import torch
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
    
    
def 
    