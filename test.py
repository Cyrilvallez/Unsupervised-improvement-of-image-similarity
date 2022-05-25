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

path = 'Datasets/Flickr500K/0/'

ref = Image.open(path + f'{0}.jpg')
target = Image.open(path + f'{123}.jpg')
neighbors = [Image.open(path + f'{index}.jpg') for index in range(1, 2)]

images = (ref, neighbors, target)

out = utils.concatenate_images(images)
out.save('test.png')