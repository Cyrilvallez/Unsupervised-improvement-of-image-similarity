#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:20:16 2022

@author: cyrilvallez
"""

import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output, tensor)
        output = torch.cat(output, 0)
        
        return output

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out