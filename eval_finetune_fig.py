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



epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

diameters = []
centroids = []
dist_to_centroids = []
min_to_centroids = []


folder = 'Clustering_results/clean_dataset/euclidean_DBSCAN_SimCLR_v2_ResNet50_2x_20_samples'
features, _ = tools.extract_features_from_folder_name(folder)
assignments = tools.get_groundtruth_attribute(folder, 'assignment')

diameters.append(tools.get_groundtruth_attribute(folder,'diameters', quantile=1.))
centroids.append(tools.get_groundtruth_attribute(folder,'centroids'))

dist_to_centroid_ = []
for i, cluster in enumerate(np.unique(assignments)):
    indices = np.argwhere(assignments == cluster).flatten()
    mean_dist = np.mean(np.linalg.norm(features[indices] - centroids[0][i], axis=1))
    dist_to_centroid_.append(mean_dist)
    
dist_to_centroids.append(np.array(dist_to_centroid_))

minimums = []
for i, cluster in enumerate(np.unique(assignments)):
    indices = assignments != cluster
    min_ = np.min(np.linalg.norm(features[indices] - centroids[0][i, :], axis=1))
    minimums.append(min_)
    
min_to_centroids.append(np.array(minimums))

for epoch in epochs[1:]:
    
    diameters.append(np.load(f'Finetuning_eval/diameters_epochs_{epoch}.npy'))
    centroids.append(np.load(f'Finetuning_eval/centroids_epochs_{epoch}.npy'))
    dist_to_centroids.append(np.load(f'Finetuning_eval/mean_dist_inside_cluster_epochs_{epoch}.npy'))
    min_to_centroids.append(np.load(f'Finetuning_eval/min_dist_outside_cluster_epochs_{epoch}.npy'))
    
mean_diameters = [np.mean(a) for a in diameters]


# centroid to centroid distances
centroid_to_centroid = []
for i, epoch in enumerate(epochs):
    distances = []
    for j in range(len(centroids[i])):
        current_centroid = centroids[i][j]
        dist = np.linalg.norm(np.delete(centroids[i], j, axis=0) - current_centroid, axis=1)
        distances.append(np.mean(dist))
    centroid_to_centroid.append(np.array(distances))
    
    


    
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

plt.figure(figsize=[0.7*6.4, 0.7*4.8])
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




# Points to centroids distances
distances = np.concatenate(dist_to_centroids)
frame2 = pd.DataFrame({'Epoch': N_epochs, 'Mean point to centroid distance': distances})

plt.figure(figsize=[0.7*6.4, 0.7*4.8])
sns.violinplot(x='Epoch', y='Mean point to centroid distance', data=frame2, palette=palette)
# _, metric, _ = tools.extract_params_from_folder_name(directory)
# plt.ylabel(f'Cluster diameter ({metric} distance)')

locs, labels = plt.xticks()
labels = [label.get_text().split()[0] for label in labels]
plt.xticks(locs, labels)
legend_elements = [Patch(facecolor='tab:blue', label='Finetuning'),
                    Patch(facecolor='tab:red', label='Original')]
plt.legend(handles=legend_elements, loc='best')
plt.savefig('Distances_to_centroid.pdf', bbox_inches='tight')


# centroid to centroid distances
distances3 = np.concatenate(centroid_to_centroid)
frame3 = pd.DataFrame({'Epoch': N_epochs, 'Mean centroid to centroid distance': distances3})

plt.figure(figsize=[0.7*6.4, 0.7*4.8])
sns.violinplot(x='Epoch', y='Mean centroid to centroid distance', data=frame3, palette=palette)
# _, metric, _ = tools.extract_params_from_folder_name(directory)
# plt.ylabel(f'Cluster diameter ({metric} distance)')

locs, labels = plt.xticks()
labels = [label.get_text().split()[0] for label in labels]
plt.xticks(locs, labels)
legend_elements = [Patch(facecolor='tab:blue', label='Finetuning'),
                    Patch(facecolor='tab:red', label='Original')]
plt.legend(handles=legend_elements, loc='best')
plt.savefig('Centroid_to_centroid_distance.pdf', bbox_inches='tight')


# min points to centroid distances
distances4 = np.concatenate(min_to_centroids)
frame4 = pd.DataFrame({'Epoch': N_epochs, 'Min outside point to centroid distance': distances4})

plt.figure(figsize=[0.7*6.4, 0.7*4.8])
sns.violinplot(x='Epoch', y='Min outside point to centroid distance', data=frame4, palette=palette)
# _, metric, _ = tools.extract_params_from_folder_name(directory)
# plt.ylabel(f'Cluster diameter ({metric} distance)')

locs, labels = plt.xticks()
labels = [label.get_text().split()[0] for label in labels]
plt.xticks(locs, labels)
legend_elements = [Patch(facecolor='tab:blue', label='Finetuning'),
                    Patch(facecolor='tab:red', label='Original')]
plt.legend(handles=legend_elements, loc='best')
plt.savefig('Min_points_to_centroid.pdf', bbox_inches='tight')





