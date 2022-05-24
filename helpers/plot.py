#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:54:40 2022

@author: cyrilvallez
"""
import matplotlib.pyplot as plt
import numpy as np

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

def time_recall_plot_flat(results, save=False, filename=None):
    
    recall = []
    searching_time = []
    metric = []
    k = []
    
    for key in results.keys():
        metric.append(key.rsplit('--', 1)[1])
        recall.append(results[key]['recall'])
        k.append(results[key]['k'])
        searching_time.append(results[key]['searching_time'])
        
    assert(np.array_equal(k, len(k)*[k[0]]))
        
    
    plt.figure()
    for i in range(len(recall)):
        plt.scatter(recall[i], searching_time[i], color=COLORS[metric[i]], 
                    marker=MARKERS[metric[i]], label=metric[i])

    plt.xlabel(f'Recall@{k[0]}')
    plt.ylabel('Search time [s]')
    plt.legend()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
def time_recall_plot_flat_sweep_k(results, save=False, filename=None):
    
    recall = []
    searching_time = []
    metric = []
    k = []
    
    for key in results.keys():
        metric.append(key.rsplit('--', 1)[1])
        recall.append(results[key]['recall'])
        k.append(results[key]['k'])
        searching_time.append(results[key]['searching_time'])
    """
    plt.figure()
    for i in range(len(recall)):
        plt.scatter(recall[i], searching_time[i], color=COLORS[metric[i]], 
                    marker=MARKERS[metric[i]], label=metric[i])
        for j in range(len(recall[i])):
            plt.annotate(f'k={k[i][j]}', (recall[i][j], searching_time[i][j]),
                         xytext=(3,3), textcoords='offset pixels')

    plt.xlabel('Recall@k')
    plt.ylabel('Search time [s]')
    plt.legend()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    """
    
    plt.figure()
    for i in range(len(recall)):
        plt.scatter(k[i], recall[i], color=COLORS[metric[i]], 
                    marker=MARKERS[metric[i]], label=metric[i])
    plt.xlabel('k')
    plt.ylabel('Recall@k')
    plt.legend()
    if save:
        plt.savefig(filename + 'recall.pdf', bbox_inches='tight')
    plt.show()
        
    plt.figure()
    for i in range(len(recall)):
        plt.scatter(k[i], searching_time[i], color=COLORS[metric[i]], 
                    marker=MARKERS[metric[i]], label=metric[i])
    plt.xlabel('k')
    plt.ylabel('Search time [s]')
    plt.legend()
    if save:
        plt.savefig(filename + 'time.pdf', bbox_inches='tight')
    plt.show()
    
def time_recall_plot_IVF(results, ylog=False, save=False, filename=None):
    
    recall = []
    searching_time = []
    metric = []
    k = []
    method = []
    
    for key in results.keys():
        method.append(key.split('--', 1)[0])
        metric.append(key.split('--', 1)[1])
        recall.append(results[key]['recall'])
        k.append(results[key]['k'])
        searching_time.append(results[key]['searching_time'])
        
    assert(np.array_equal(k, len(k)*[k[0]]))
        
    
    plt.figure()
    for i in range(len(recall)):
        if 'IVF' in method[i]:
            plt.plot(recall[i], searching_time[i], color=COLORS[metric[i]], 
                     label=method[i] + '-' + metric[i])
        else:
            plt.scatter(recall[i], searching_time[i], color=COLORS[metric[i]], 
                        marker=MARKERS[metric[i]], label=method[i] + '-' + metric[i])

    plt.xlabel(f'Recall@{k[0]}')
    plt.ylabel('Search time [s]')
    plt.legend()
    if ylog:
        plt.yscale('log')
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
