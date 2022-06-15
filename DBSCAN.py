#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:46:36 2022

@author: cyrilvallez
"""

import argparse

from clustering.clustering import cluster_DBSCAN
            
            
# Parsing of command line args
parser = argparse.ArgumentParser(description='Clustering of the memes')
parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                    help='The algorithm from which the features describing the images derive.')
parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                    help='The metric for distance between features.')
parser.add_argument('--samples', type=int, default=5, 
                    help='The number of samples in a neighborhood for a point to be considered as a core point.')
args = parser.parse_args()

algorithm = ' '.join(args.algo)
metric = args.metric
min_samples = args.samples
    
# Clustering
cluster_DBSCAN(algorithm, metric, min_samples)
        
        
        
        
        