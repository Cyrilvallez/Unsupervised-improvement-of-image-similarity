#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:57:01 2022

@author: cyrilvallez
"""

import numpy as np
import faiss
from helpers import utils
import experiment as ex
import os


algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'Kaggle_templates'
distractor_dataset = 'Flickr500K'
query_dataset = 'Kaggle_memes'

factory_str = 'Flat'

experiment = ex.Experiment(algorithm, main_dataset, query_dataset,
                        distractor_dataset=distractor_dataset)

experiment.set_index(factory_str, metric='cosine')
experiment.fit(k=10)

save_folder = 'Results/Neighbors_memes_flat_cosine/'
dirname = os.path.dirname(save_folder)
exist = os.path.exists(dirname)
if not exist:
    os.makedirs(dirname)



for index in range(len(experiment.mapping_query)):

    ref, neighbors = experiment.get_neighbors_of_query(index)
    neighbors = utils.concatenate_images(ref, neighbors)
    neighbors.save(save_folder + f'{index}.png')