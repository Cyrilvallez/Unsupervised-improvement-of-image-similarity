#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:03 2022

@author: cyrilvallez
"""
"""
import faiss
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from helpers import utils
import experiment as ex

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
# save_folder = utils.parse_input()

algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'BSDS500_original'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'

# factory_str = 'Flat'
metrics = ['L2', 'L1', 'cosine']

# experiment = ex.Experiment(factory_str, algorithm, main_dataset, query_dataset,
                           # distractor_dataset=distractor_dataset, metric=metrics[0])

# nlist = int(10*np.sqrt(features_db.shape[0]))

filename = save_folder + 'results.json'
ex.compare_metrics_Flat(metrics, algorithm, main_dataset, query_dataset,
                        distractor_dataset, filename, k=1)

"""
#%%

import faiss
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from helpers import utils
import experiment as ex


save_folder = utils.parse_input()

algorithm = 'SimCLR v2 ResNet50 2x'
main_dataset = 'BSDS500_original'
distractor_dataset = 'Flickr500K'
query_dataset = 'BSDS500_attacks'

features_db, mapping_db = utils.combine_features(algorithm, main_dataset,
                                                           distractor_dataset)
features_query, mapping_query = utils.load_features(algorithm, query_dataset)

search_identifiers = np.array([name.rsplit('/', 1)[1].split('_', 1)[0] for name in mapping_query])
db_identifiers = np.array([name.rsplit('/', 1)[1].rsplit('.', 1)[0] for name in mapping_db])

print(search_identifiers[0:50])
print('\n \n \n')
print(db_identifiers[0:50])