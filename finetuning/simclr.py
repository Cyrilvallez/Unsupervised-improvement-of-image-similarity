#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:28:09 2022

@author: cyrilvallez
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from finetuning.loss import NT_Xent
from finetuning.lars import LARS


class SimCLR(nn.Module):
    
    def __init__(self, encoder, contrastive_head):
        
        super().__init__()
        self.encoder = encoder
        self.contrastive_head = contrastive_head
        
    def forward(self, x1, x2):
    
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
    
        z1 = self.contrastive_head(h1)
        z2 = self.contrastive_head(h2)
        
        return z1, z2
    
    
    def configure_optimizers(self):
        
        # This is correct to give all params to the optimizer even if
        # some will be frozen during fine-tuning
        params = self.parameters()
            
        if self.optimizer == 'lars':
            optimizer = LARS(params, lr=self.learning_rate, momentum=self.momemtum,
                             weight_decay=self.weight_decay, nesterov=self.nesterov)
            
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        
        if self.scheduler is not None:
            scheduler = {'scheduler': self.scheduler, 'interval':self.lr_interval,
                         'frequency':self.lr_frequency}
        
            return {'optimizer': optimizer, 'lr_scheduler':scheduler}
        
        else:
            return optimizer
        
        
    def save(self, path):
        
        state = {'encoder': self.encoder.state_dict(),
                 'head': self.contrastive_head.state_dict()}
        torch.save(state, path)
        
    @staticmethod
    def load(path):
        
        encoder, head = SIMv2.get_resnet(depth=50, width_multiplier=2,
                                        sk_ratio=0.0625)
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder'])
        head.load_state_dict(checkpoint['head'])
        
        return SimCLR(encoder, head)


def train_one_epoch(model, train_dataloader, criterion, optimizer,
                    rank, distributed):
        
    running_average_loss = 0.
    
    for step, batch in enumerate(train_dataloader):
        
        # Get batch and set devices
        x1, x2 = batch
        x1 = x1.cuda(rank)
        x2 = x2.cuda(rank)
        
        # clean the gradients
        optimizer.zero_grad()
        
        # Forward pass
        h1, h2 = model(x1, x2)
        
        # Compute the loss and backward pass
        loss = criterion(h1, h2)
        loss.backward()
        
        # Adjust the weights
        optimizer.step()
        
        # Update the running average
        running_average_loss += loss.item()
        
    # Get the average of the loss and converts to tensor
    running_average_loss /= (step+1)
    running_average_loss = torch.tensor(running_average_loss, dtype=loss.dtype,
                                        device=loss.device)
        
    # Get average of running losses if distributed training
    if distributed:
        dist.all_reduce(running_average_loss, op=dist.ReduceOp.AVG, async_op=False)
        
    return running_average_loss.item()


def validate_one_epoch(model, val_dataloader, criterion, rank=0):
    
    val_loss = 0.
    
    with torch.no_grad():
        
        for step, batch in enumerate(val_dataloader):
            
            # Get batch and set devices
            x1, x2 = batch
            x1 = x1.cuda(rank)
            x2 = x2.cuda(rank)
            
            # Forward pass
            h1, h2 = model(x1, x2)
            
            # Compute the loss and backward pass
            loss = criterion(h1, h2)
            val_loss += loss.item()
            
        # Get the average of the loss 
        val_loss /= (step+1)
            
        return val_loss
        

def train(epochs, model, train_dataloader, val_dataloader, criterion, optimizer,
          scheduler, writer):
    
    if dist.is_available() and dist.is_initialized():
        distributed = True
        rank = dist.get_rank()
    else:
        distributed = False
        rank = 0
        
    # We need to compute the gradients
    torch.set_grad_enabled(True)
    
    for epoch in tqdm(range(epochs)):
        
        # Train
        model.train(True)
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer,
                                     rank, distributed)
        
        # Get actual learning rate
        if scheduler is not None:
            lr = scheduler.get_last_lr()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Eventually validate (but only on one GPU in case of distributed training)
        if val_dataloader is not None:
            if rank == 0:
                model.eval()
                val_loss = validate_one_epoch(model, val_dataloader, criterion,
                                              rank)
                
        # Print a summary of current epoch
        if rank == 0:
            if val_dataloader is not None:
                print(f'Epoch {epoch} --- train_loss : {train_loss:.3f}, val_loss : {val_loss:.3f}')
            else:
                print(f'Epoch {epoch} --- train_loss : {train_loss:.3f}')
                   
        # Log quantities
        if rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            if val_dataloader is not None:
                writer.add_scalar('Loss/validation', val_loss, epoch)
            if scheduler is not None:
                writer.add_scalar("Misc/learning_rate", lr, epoch)
                
        # Save the actual model
        if rank == 0:
            model.save(path)
        
        # Synchonize processes as process on rank 0 should have an overhead
        # from validation, saving etc...
        if distributed:
            dist.barrier()
        
        
#%%

from extractor.SimCLRv2 import resnet as SIMv2

import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    
writer.add_hparams({'lr': 0.1*np.random.random(), 'bsize': np.random.random()},
                   {'test':2})

