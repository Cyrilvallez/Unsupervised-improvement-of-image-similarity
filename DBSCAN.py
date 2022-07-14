#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:46:36 2022

@author: cyrilvallez
"""

import argparse
import numpy as np

from clustering.clustering import cluster_DBSCAN, save_all
            
            
# Parsing of command line args
parser = argparse.ArgumentParser(description='Clustering of the memes')
parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                    help='The algorithm from which the features describing the images derive.')
parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                    help='The metric for distance between features.')
parser.add_argument('--precisions', type=float, nargs='+', default=None,
                    help=('The maximum distance between two samples for one to be'
                          'considered as in the neighborhood of the other.'))
parser.add_argument('--samples', type=int, default=5, 
                    help='The number of samples in a neighborhood for a point to be considered as a core point.')
parser.add_argument('--partition', type=str, default='full', choices=['full', 'clean'],
                    help='Dataset partition to use.')
parser.add_argument('--save', type=str, default='True', choices=['True', 'False'],
                    help='Whether to save everything, or just the cluster assignments.')
args = parser.parse_args()

algorithm = ' '.join(args.algo)
metric = args.metric
min_samples = args.samples
precisions = args.precisions
partition = args.partition + '_dataset'
save = args.save == 'True'

# precisions = np.linspace(0.01, 7, 15)
    
# Clustering
directory = cluster_DBSCAN(algorithm, metric, min_samples, precisions, partition)
if save:
    save_all(directory)
        
        