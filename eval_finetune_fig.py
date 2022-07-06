#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:14:30 2022

@author: cyrilvallez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd

from clustering import tools
from helpers import plot_config

epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80]

diameters = []
centroids = []

diameters.append(tools.get_groundtruth_attribute('Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples',
                                                 'diameters', quantile=1.))
centroids.append(tools.get_groundtruth_attribute('Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples',
                                                 'centroids'))

for epoch in epochs[1:]:
    
    diameters.append(np.load(f'Finetuning_eval/diameters_epochs_{epoch}.npy'))
    centroids.append(np.load(f'Finetuning_eval/centroids_epochs_{epoch}.npy'))
    
mean_diameters = [np.mean(a) for a in diameters]
    
plt.figure()
plt.plot(epochs, mean_diameters)
plt.xlabel('Epoch')
plt.ylabel('Mean cluster diameter')
plt.grid()


N_epochs = [epochs[i]*np.ones(len(diameters[i]), dtype=int) \
               for i in range(len(diameters))]
N_epochs = np.concatenate(N_epochs)
diameters = np.concatenate(diameters)

palette = ['tab:red' if i == 0 else 'tab:blue' for i in range(len(epochs))]
frame = pd.DataFrame({'Epoch': N_epochs, 'Cluster diameter': diameters})

plt.figure(figsize=(8,8))
sns.violinplot(x='Epoch', y='Cluster diameter', data=frame, palette=palette)
# _, metric, _ = tools.extract_params_from_folder_name(directory)
# plt.ylabel(f'Cluster diameter ({metric} distance)')

locs, labels = plt.xticks()
labels = [label.get_text().split()[0] for label in labels]
plt.xticks(locs, labels)
legend_elements = [Patch(facecolor='tab:blue', label='Finetuning'),
                    Patch(facecolor='tab:red', label='Original')]
plt.legend(handles=legend_elements, loc='best')
plt.savefig('Diameters_finetuning.pdf', bbox_inches='tight')


