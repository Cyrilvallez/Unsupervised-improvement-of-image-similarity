#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss

d = 2048                         # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

#%%
import numpy as np
import faiss
from matplotlib import pyplot

# generate data

# different variance in each dimension
x0 = faiss.randn((1000, 16)) * (1.2 ** -np.arange(16))

# random rotation
R, _ = np.linalg.qr(faiss.randn((16, 16)))   
x = np.dot(x0, R).astype('float32')

# compute and visualize the covariance matrix
xc = x - x.mean(0)
cov = np.dot(xc.T, xc) / xc.shape[0]
_ = pyplot.imshow(cov)

