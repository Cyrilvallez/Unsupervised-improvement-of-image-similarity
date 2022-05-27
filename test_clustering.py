#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:28:49 2022

@author: cyrilvallez
"""

import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

distances = np.load('distances.npy')
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                     affinity='precomputed', linkage='average')

t0 = time.time()

clustering.fit(distances)

dt = time.time() - t0

def plot_dendrogram(model, distances, **kwargs):
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
    dendrogram(linkage_matrix, **kwargs)
    
   
#%%

# plot the top three levels of the dendrogram
plot_dendrogram(clustering, distances, truncate_mode="level", p=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
