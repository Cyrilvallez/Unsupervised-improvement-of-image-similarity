#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:57:01 2022

@author: cyrilvallez
"""

import os

from helpers import utils
from fast_search import search


algorithm = 'SimCLR v2 ResNet50 2x'
# algorithm = 'Dhash 64 bits'
# main_dataset = 'BSDS500_original'
main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
# query_dataset = 'BSDS500_attacks'
query_dataset = 'Kaggle_memes'

save_folder = 'Results/Contrastive_memes_L2/'

factory_str = 'Flat'
metric = 'L2'

experiment = search.Experiment(algorithm, main_dataset, query_dataset,
                        distractor_dataset=distractor_dataset)

experiment.set_index(factory_str, metric=metric)
experiment.to_gpu()
experiment.fit(k=10)
_, correct = experiment.recall()


if save_folder[-1] != '/':
    save_folder = '/'
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_folder + 'Correct', exist_ok=True)
os.makedirs(save_folder + 'Incorrect', exist_ok=True)


for index, status in enumerate(correct):

    images = experiment.get_neighbors_of_query(index)
    neighbors = utils.concatenate_images(images)
    if status == 1:
        neighbors.save(save_folder + f'Correct/{index}.png')
    elif status == 0:
        neighbors.save(save_folder + f'Incorrect/{index}.png')
    
