#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:53:54 2022

@author: cyrilvallez
"""
from tqdm import tqdm
import os

from clustering import tools

# directory = 'Clustering_results'
# for folder in [f.path for f in os.scandir(directory) if f.is_dir()]:
    # counts = cluster_size_violin(folder, save=True, filename='sizes_violin.pdf')
    # cluster_size_evolution(folder, save=False, filename='sizes.pdf')
    # cluster_diameter_violin(folder, save=True, filename='sizes_violin.pdf')
        
directory = 'Clustering_results'
for folder in tqdm([f.path for f in os.scandir(directory) if f.is_dir()]):
    tools.save_diameters(folder)
    

     