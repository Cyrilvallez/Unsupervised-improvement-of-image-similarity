#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import faiss


features = np.random.choice(2, size=(500000, 64))
query = np.random.choice(2, size=(2000, 64))

index = faiss.IndexBinaryFlat(features.shape[1])
index = faiss.index_cpu_to_all_gpus(index)

index.add(features)
D, I = index.search(query, 10)



