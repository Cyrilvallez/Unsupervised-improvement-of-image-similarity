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
        
        # reduce is needed since the loss will be computed only for the 
        # batch of each process, and not directly using the gathered output
        # (this is coherent with PyTorch structure)
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
    
    def __init__(self, temperature, epsilon=1e-7):
        
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, z1, z2):
        """
        Compute the NT-Xent loss. Note that in a distributed training framework,
        this only computes the loss with respect to the batch seen by each process.
        Thus the `all_reduce` operation is needed in the backward of `gather`
        function.

        Parameters
        ----------
        z1 : Tensor
            Model output corresponding to the first data augmentation.
        z2 : Tensor
            Model output corresponding to the second data augmentation.

        Returns
        -------
        loss : float (as a Tensor)
            The loss value.

        """
        
        # Normalize the vectors. Use max(norm, epsilon) for numerical stability
        norm1 = torch.linalg.norm(z1, ord=2, dim=1)
        norm2 = torch.linalg.norm(z2, ord=2, dim=1)
        epsilon = self.epsilon*torch.ones(z1.shape[0],1, dtype=z1.dtype,
                                          device=z1.device)
        z1 = z1 / torch.max(norm1.unsqueeze(1), epsilon)
        z2 = z2 / torch.max(norm2.unsqueeze(1), epsilon)
        
        # Gather batch from all processes
        if dist.is_available() and dist.is_initialized():
            z1_tot = gather(z1)
            z2_tot = gather(z2)
        else:
            z1_tot = z1
            z2_tot = z2
        
        # Concatenate to make use of matrix multiplication
        z_tot = torch.cat((z1_tot, z2_tot), dim=0)
        z = torch.cat((z1, z2), dim=0)
        
        # Compute pairwise cosine similarity
        similarity = torch.mm(z, z_tot.t()) 
        sim_exp = torch.exp(similarity / self.temperature)
        
        # Compute the denominator of the loss. See paper of SimCLRv1.
        # We need to remove the unit similarity term corresponding to s_i_i
        denominator = torch.sum(sim_exp, dim=1) - \
            torch.exp(torch.tensor(1./self.temperature, dtype=z.dtype, device=z.device))
        
        # Compute cosine similarity for the positives examples 
        positives = torch.sum(z1 * z2, dim=1)
        positives = torch.exp(positives / self.temperature)
        # Cosine similarity is symetric, thus s_i_j = s_j_i
        positives = torch.cat((positives, positives), dim=0)
        
        loss = torch.mean(-torch.log(positives / denominator))
        
        return loss
    



     



# =============================================================================
# 
# =============================================================================

class GatherLayer(torch.autograd.Function):
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
    
    
    
    
    
class NT_Xent2(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

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

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

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
    