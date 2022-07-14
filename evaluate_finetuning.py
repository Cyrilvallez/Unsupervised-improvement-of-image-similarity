#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:58:28 2022

@author: cyrilvallez
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd

import extractor
from finetuning.simclr import SimCLR
from clustering import metrics
from clustering import tools


def density_plot(epochs, metric, ylabel, filename):
    
    N_epochs = [epochs[i]*np.ones(len(metric[i]), dtype=int) \
                   for i in range(len(metric))]
    N_epochs = np.concatenate(N_epochs)
    metric = np.concatenate(metric)

    palette = ['tab:red' if i == 0 else 'tab:blue' for i in range(len(epochs))]
    frame = pd.DataFrame({'Epoch': N_epochs, 'metric': metric})

    plt.figure()
    sns.violinplot(x='Epoch', y='metric', data=frame, palette=palette)
    plt.ylabel(ylabel)
    legend_elements = [Patch(facecolor='tab:blue', label='Finetuning'),
                        Patch(facecolor='tab:red', label='Original')]
    plt.legend(handles=legend_elements, loc='best')
    plt.savefig(filename, bbox_inches='tight')




transforms = extractor.SIMCLR_TRANSFORMS
dataset = extractor.create_dataset('all_memes', transforms)

epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
# Paths where the finetuned models have been saved
paths = [f'first_test_models/2022-06-30_20:03:28/epoch_{a}.pth' for a in epochs[1:]]
    
all_diameters = []
all_centroids = []
all_separations = []
all_dispersions = []
all_centroid_to_centroid = []
all_signal_noise_ratio1 = []
all_signal_noise_ratio2 = []

for epoch, path in zip(epochs, paths):
        
    if epoch == 0:
        features, mapping = tools.get_features('clean_dataset', 'SimCLR v2 ResNet50 2x')
    else:
        # Load model
        model = SimCLR.load_encoder(path)
        # Give the model a name
        name = f'SimCLR_finetuned_epoch_{epoch}'
        # Save features
        extractor.extract_and_save_neural(model, dataset, name=name)
        # Load the features of only the perceptually similar memes
        features, mapping = tools.get_features('clean_dataset', name)
    
    assignments = tools.compute_assignment_groundtruth(mapping)

    diameters = metrics.cluster_diameters(features, assignments, quantile=1.)
    all_diameters.append(diameters)
        
    centroids = metrics.cluster_centroids(features, assignments)
    all_centroids.append(centroids)
    
    all_separations.append(metrics.outside_cluster_separation(features, assignments,
                                                              centroids))
    all_dispersions.append(metrics.inside_cluster_dispersion(features, assignments,
                                                             centroids))
    all_centroid_to_centroid.append(metrics.mean_centroid_to_centroid(features, assignments,
                                                                      centroids))
    all_signal_noise_ratio1.append(metrics.diameters_over_separations(features, assignments,
                                                                      diameters, centroids))
    all_signal_noise_ratio2.append(metrics.dispersion_over_centroid_to_centroid(features, assignments,
                                                                                centroids))
    
    
folder = 'Finetuning_eval_fig'
os.makedirs(folder, exist_ok=True)

density_plot(epochs, all_diameters, filename=folder + 'diameters.pdf',
             ylabel='Diameters')
density_plot(epochs, all_separations, filename=folder + 'separations.pdf',
             ylabel='Min centroid to outside points distance')
density_plot(epochs, all_dispersions, filename=folder + 'dispersions.pdf',
             ylabel='Mean centroid to inside points distance')
density_plot(epochs, all_centroid_to_centroid, filename=folder + 'centroid_to_centroid.pdf',
             ylabel='Mean centroid to centroid distance')
density_plot(epochs, all_signal_noise_ratio1, filename=folder + 'SN_ratio1.pdf',
             ylabel='Diameter over min centroid to outside points distance')
density_plot(epochs, all_signal_noise_ratio2, filename=folder + 'SN_ratio2.pdf',
             ylabel='Mean centroid to inside points distance over centroid to centroid distance')
        
 


