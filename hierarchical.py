#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:48:37 2022

@author: cyrilvallez
"""

import argparse

from clustering.clustering import hierarchical_clustering, save_all

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Clustering of the memes')

parser.add_argument('--algo', type=str, nargs='+', default='SimCLR v2 ResNet50 2x'.split(),
                    help='The algorithm from which the features describing the images derive.')

parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                    help='The metric for distance between features.')

parser.add_argument('--linkage', type=str, default='ward',
                    choices=['single', 'complete', 'average', 'centroid', 'ward'],
                    help='The linkage method for merging clusters.')

parser.add_argument('--partition', type=str, default='full', choices=['full', 'clean'],
                    help='Dataset partition to use.')

parser.add_argument('--save', type=str, default='True', choices=['True', 'False'],
                    help='Whether to save everything, or just the cluster assignments.')

args = parser.parse_args()


algorithm = ' '.join(args.algo)
metric = args.metric
linkage_type = args.linkage
partition = args.partition + '_dataset'
save = args.save == 'True'
    
# Clustering
directory = hierarchical_clustering(algorithm, metric, linkage_type, partition)
if save:
    save_all(directory)