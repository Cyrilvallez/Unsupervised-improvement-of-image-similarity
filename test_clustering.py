#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:28:49 2022

@author: cyrilvallez
"""

import numpy as np
import time
import sklearn
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

distances = np.load('distances_neighbors.npy')
connectivity = np.zeros(distances.shape)
connectivity[distances != 0] = 1

clustering = AgglomerativeClustering(n_clusters=250, distance_threshold=None,
                                     affinity='precomputed', linkage='single')

t0 = time.time()

clustering.fit(distances)

dt = time.time() - t0

#%%

def get_dendrogram(model, truncate_mode='level', p=10, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p, **kwargs)
    

# plot the top three levels of the dendrogram
# plot_dendrogram(clustering)
# plt.title("Hierarchical Clustering Dendrogram")
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()





#%%

dendro = get_dendrogram(clustering)


fig = plt.figure(figsize=(8,8))
ax_dendrogram = fig.add_axes([0., 0.71, 0.9, 0.3])
ax_heatmap = fig.add_axes([0., 0., 0.9, 0.7])

ax_dendrogram.set_xticks([])
ax_dendrogram.set_yticks([])
ax_heatmap.set_xticks([])
ax_heatmap.set_yticks([])

# Plot distance matrix.
# idx = dendro['leaves']
# distances = distances[idx,idx]
im = ax_heatmap.matshow(distances, aspect='auto', origin='lower')

# Plot colorbar
ax_color = fig.add_axes([0.91, 0., 0.02, 0.7])
plt.colorbar(im, cax=ax_color)
fig.show()
# fig.savefig('dendrogram.png')





# =============================================================================
#
# =============================================================================
#%%

from helpers import utils
import scipy.cluster.hierarchy as hierarchy

algorithm = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_memes'

features, _ = utils.load_features(algorithm, dataset)
Z = hierarchy.linkage(features, method='ward')

plt.figure(figsize=(10,10))
Z2 = hierarchy.dendrogram(Z)
ticks = plt.xticks()
plt.xticks(np.arange(0, 43000, 2000))
plt.xlabel('')
plt.ylabel('Euclidean distance')