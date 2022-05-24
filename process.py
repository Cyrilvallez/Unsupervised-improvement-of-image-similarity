#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:12:07 2022

@author: cyrilvallez
"""

# =============================================================================
# File to process and plot results from experiments
# =============================================================================

import os
from helpers import utils
from helpers import plot

EXPERIMENT_NAME = 'Flat_memes_sweep_k'



if EXPERIMENT_NAME[-1] != '/':
    EXPERIMENT_NAME += '/'

experiment_folder = 'Results/' + EXPERIMENT_NAME 
experiment_name = experiment_folder + 'results.json'
   
    
    
results = utils.load_dictionary(experiment_name)


#%%

save = True

# plot.time_recall_plot_flat(results, save=save, filename=experiment_folder + 'time_recall.pdf')
plot.time_recall_plot_flat_sweep_k(results, save=save, filename=experiment_folder + 'time_recall.pdf')




        
        


