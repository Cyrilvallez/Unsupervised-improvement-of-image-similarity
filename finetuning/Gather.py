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
    
    
    
#%%

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
    
    
import os

# dist.destroy_process_group()
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo', rank=0, world_size=1)
a1 = torch.rand(2, requires_grad=True)
foo1 = test1.apply(a1)
external_grad = torch.tensor([2., 1.])
bar1 = foo1.backward(external_grad)
grad1 = a1.grad

a2 = torch.rand(2, requires_grad=True)
foo2 = test2.apply(a2)
external_grad = torch.tensor([1., 1.])
bar2 = foo2.backward(external_grad)
grad2 = a2.grad

#%%

a = torch.arange(2, requires_grad=True, dtype=float)
b = torch.tensor([3.,4], requires_grad=True)

c = a + b
d = 2*a - b

Q = c**2 - d

external_grad = torch.tensor([1., 1.])
foo = Q.backward(gradient=external_grad)
#%%
a = torch.arange(2, requires_grad=True, dtype=float)
b = torch.tensor([3.,4], requires_grad=True)

#%%
c = torch.tensor([3., 5.],requires_grad=True)
d = torch.tensor([-3., -2.],requires_grad=True)

Q = c**2 - d

external_grad = torch.tensor([1., 1.])
foo = Q.backward(gradient=external_grad)



#%%

def foo(*args):
    return args[2]

a = foo(0,1,19,3)
print(a)
    