#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:54:40 2022

@author: cyrilvallez
"""
import matplotlib.pyplot as plt

from helpers import plot_config

COLORS = {
    'cosine': 'b',
    'L2': 'r',
    'L1': 'g',
    }

MARKERS = {
    'cosine': '*',
    'L2': 'o',
    'L1': 'x',
    }

def time_recall_plot(results, save=False, filename=None):
    
    recall = []
    searching_time = []
    metric = []
    
    for key in results.keys():
        metric.append(key.rsplit('--', 1)[1])
        # recall.append(results[key])
        searching_time.append(results[key]['searching_time'])
        
        for subkey in results[key].keys():
            if 'recall' in subkey:
                xlabel = subkey
                recall.append(results[key][subkey])
    
    plt.figure()
    for i in range(len(names)):
        if names[i] == 'JS Flat':
            plt.scatter(recalls[i], times[i], color='r', marker='*', s=100)
        if names[i] == 'cosine Flat':
            plt.scatter(recalls[i], times[i], color='b', marker='o', s=100)
        if names[i] == 'L2 Flat':
            plt.scatter(recalls[i], times[i], color='g', marker='x', s=100)
        if names[i] == 'L1 Flat':
            plt.scatter(recalls[i], times[i], color='m', marker='s', s=100)
        if names[i] == 'JS IVFFlat':
            plt.plot(recalls[i], times[i], color='r')
        if names[i] == 'cosine IVFFlat':
            plt.plot(recalls[i], times[i], color='b')
        if names[i] == 'L2 IVFFlat':
            plt.plot(recalls[i], times[i], color='g')
        if names[i] == 'L1 IVFFlat':
            plt.plot(recalls[i], times[i], color='m')

    plt.xlabel('Recall@1')
    plt.ylabel('Search time [s]')
    plt.legend(names)
        
    plt.savefig('benchmark.pdf', bbox_inches='tight')
    plt.show()
