#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:11 2022

@author: cyrilvallez
"""

import numpy as np
import shutil
from PIL import Image
from helpers import utils
import torch

algorithm = 'SimCLR v2 ResNet50 2x'
dataset = 'Kaggle_memes'

features = utils.load_features(algorithm, dataset)
features = torch.tensor(features).to('cuda')

res = torch.cdist(features, features).cpu().numpy()
np.save('distances.npy', res)
