#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""

import faiss
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from helpers import utils
import experiment as ex

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()

algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
query_dataset = 'Kaggle_memes'

# factory_str = 'Flat'
metrics = ['L2', 'L1', 'cosine']

# experiment = ex.Experiment(factory_str, algorithm, main_dataset, query_dataset,
                           # distractor_dataset=distractor_dataset, metric=metrics[0])

# nlist = int(10*np.sqrt(features_db.shape[0]))

filename = save_folder + 'results.json'
ex.compare_metrics_Flat(metrics, algorithm, main_dataset, query_dataset,
                        distractor_dataset, filename, k=1)

