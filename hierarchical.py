#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:48:37 2022

@author: cyrilvallez
"""

import argparse

from clustering.clustering import hierarchical_clustering

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Clustering of the memes')
parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                    help='The algorithm from which the features describing the images derive.')
parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                    help='The metric for distance between features.')
parser.add_argument('--linkage', type=str, default='ward', 
                    choices=['single', 'complete', 'average', 'centroid', 'ward'],
                    help='The linkage method for merging clusters.')
args = parser.parse_args()

algorithm = ' '.join(args.algo)
metric = args.metric
linkage_type = args.linkage
    
# Clustering
hierarchical_clustering(algorithm, metric, linkage_type)