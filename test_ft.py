#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:25:46 2022

@author: cyrilvallez
"""
from torch.utils.data import DataLoader

from finetuning.training import train_one_epoch, get_optimizer
from finetuning.simclr import SimCLR
from finetuning.nt_xent import NT_Xent
from finetuning.transforms import ImageDataset

workers = 8
batch_size = 64


model = SimCLR.load()
model.train(True)
model.cuda(0)

dataset = ImageDataset('Datasets/cartoons/TRAIN/catdog')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True, drop_last=False)
optimizer, scheduler = get_optimizer(model, 10, 'lars', 1e-2, 1e-6,
                                     0.9, False)
criterion = NT_Xent(0.1)


train_one_epoch(model, dataloader, criterion, optimizer, 0, False)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**3
print('model size: {:.3f} GB'.format(size_all_mb))

N_param = sum(p.numel() for p in model.parameters())
print('{N_param} parameters')