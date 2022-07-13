#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:54:40 2022

@author: cyrilvallez
"""
import matplotlib.pyplot as plt
import numpy as np

# Some preset rc params
from helpers import plot_config

# Color for each metric for the plots
COLORS = {
    'cosine': 'b',
    'L2': 'r',
    'L1': 'g',
    'hamming': 'b'
    }

# Marker for each metric for the plots
MARKERS = {
    'cosine': '*',
    'L2': 'o',
    'L1': 'x',
    'hamming': '*'
    }

def time_recall_plot_flat(results, save=False, filename=None):
    """
    Creates a plot with searching time against recall@k for flat indices.

    Parameters
    ----------
    results : dict
        Results of an experiment.
    save : bool, optional
        Whether to save the figure or not. The default is False.
    filename : str, optional
        Filename if `save` is True. The default is None.
        
    Raises
    ------
    ValueError
        If save is True but filename is not provided.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You need to specify a filename if you want to save the figure')
    
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
    """
    Creates a plot with searching time against recall@k for different k for 
    flat indices.

    Parameters
    ----------
    results : dict
        Results of an experiment.
    save : bool, optional
        Whether to save the figure or not. The default is False.
    filename : str, optional
        Filename if `save` is True. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is not provided.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You need to specify a filename if you want to save the figure')
    
    recall = []
    searching_time = []
    metric = []
    k = []
    
    for key in results.keys():
        metric.append(key.rsplit('--', 1)[1])
        recall.append(results[key]['recall'])
        k.append(results[key]['k'])
        searching_time.append(results[key]['searching_time'])
    
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
    
    
def time_recall_plot_IVF(results, ylog=False, save=False, filename=None):
    """
    Creates a plot with searching time against recall@k for IVF indices.

    Parameters
    ----------
    results : dict
        Results of an experiment.
    ylog : bool, optional
        Whether to put the yscale in log or not.
    save : bool, optional
        Whether to save the figure or not. The default is False.
    filename : str, optional
        Filename if `save` is True. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is not provided.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You need to specify a filename if you want to save the figure')
    
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
    
    