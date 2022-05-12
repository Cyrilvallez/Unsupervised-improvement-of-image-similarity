#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss

d = 4096                         # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries

a = np.random.rand(nb, d).astype('float32')
b = np.random.rand(nq, d).astype('float32')

res = faiss.StandardGpuResources()  

index = faiss.GpuIndexFlat(res, dims=d, metric=faiss.METRIC_JensenShannon)  
print(index.metric_type)
index.add(a)                  

k = 4                          
D, I = index.search(b, k)

print('Done !')     



