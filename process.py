#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:38:24 2022

@author: cyrilvallez
"""

import faiss
from helpers import utils
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from helpers import configs_plot

with open("recall.pickle", "rb") as fp:
    recalls = pickle.load(fp)
    
with open("time.pickle", "rb") as fp:
    times = pickle.load(fp)
    
names = [
    'JS Flat',
    'cosine Flat',
    'L2 Flat',
    'L1 Flat',
    'JS IVFFlat',
    'cosine IVFFlat',
    'L2 IVFFlat',
    'L1 IVFFlat',
    ]

plt.figure()
for i in range(len(names)):
    # if names[i] == 'JS Flat':
        # plt.scatter(recalls[i], times[i], color='g', marker='*', s=100, label=names[i])
    if names[i] == 'cosine Flat':
        plt.scatter(recalls[i], times[i], color='b', marker='o', s=100, label=names[i])
    if names[i] == 'L2 Flat':
        plt.scatter(recalls[i], times[i], color='r', marker='x', s=100, label=names[i])
    # if names[i] == 'L1 Flat':
        # plt.scatter(recalls[i], times[i], color='m', marker='s', s=100, label=names[i])
    # if names[i] == 'JS IVFFlat':
        # plt.plot(recalls[i], times[i], color='g', label=names[i])
    # if names[i] == 'cosine IVFFlat':
        # plt.plot(recalls[i], times[i], color='b', label=names[i])
    # if names[i] == 'L2 IVFFlat':
        # plt.plot(recalls[i], times[i], color='r', label=names[i])
    # if names[i] == 'L1 IVFFlat':
        # plt.plot(recalls[i], times[i], color='m', label=names[i])

plt.xlabel('Recall@1')
plt.ylabel('Search time [s]')
plt.legend()

plt.savefig('Flat.pdf', bbox_inches='tight')
plt.show()