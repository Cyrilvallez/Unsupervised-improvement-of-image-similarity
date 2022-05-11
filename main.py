#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""

import faiss
import numpy as np

features_path = 'Features/Flickr500K-SimCLR_v2_ResNet50_2x_features.npy'
map_path = 'Features/Flickr500K-SimCLR_v2_ResNet50_2x_map_to_names.npy'

features = np.load(features_path)
mapping = np.load(map_path, allow_pickle=True)