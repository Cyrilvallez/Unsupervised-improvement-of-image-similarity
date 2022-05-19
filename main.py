#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""


import numpy as np

from helpers import utils
import experiment as ex

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()

algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'BSDS500_original'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'

# factory_str = 'Flat'
# metrics = ['L2', 'L1', 'cosine']


nlist = int(10*np.sqrt(500000))
nprobes = [1, 5, 10, 20, 50, 100, 200, 300, 400]

filename = save_folder + 'results.json'
ex.compare_nprobe_IVF(nlist, nprobes, algorithm, main_dataset, query_dataset,
                        distractor_dataset, filename, k=1)

