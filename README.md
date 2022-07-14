# Feature-search-at-scale

# Pre-trained SimCLR models 

The pre-trained SimCLR models we used are not available in this repository due to their large size. To download them, please navigate to the path where you cloned the repo and run the following commands from your terminal :

```
cd path_to_repo/SimCLRv1
python3 download.py 
```

This will download the files containing the models definitions to the current folder and convert them to Pytorch. This will create 2 folders, `tf_checkpoints` and `torch_checkpoints`. The folder `tf_checkpoints` contains the Tensorflow definition of the models (directly downloaded from the [github of the authors](https://github.com/google-research/simclr)), and can be safely erased if you wish to save some disk space. The `torch_checkpoints` folder contains the translation of these models to Pytorch.

The exact same procedure will download the models for SimCLRv2 : 

```
cd path_to_repo/SimCLRv2
python3 download.py 
```

By default, this will only download one model. To download others, please have a look at the `--model` argument. If unsure what is accepted argument, you can always check the help message :

> python3 download.py -h

# Datasets

We personally used 3 datasets that can be found online. They are the [BSDS500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), [Kaggle memes dataset](https://www.kaggle.com/datasets/gmorinan/most-viewed-memes-templates-of-2018) and the first half of the [Flickr 1M dataset](https://press.liacs.nl/mirflickr/mirdownload.html). For the kaggle memes dataset, one then need to run `data_retrieval/kaggle_splitter.py` to extract templates and annotate correctly the memes. 

## BSDS500 dataset

For the BSDS500 dataset, please download it and save the images in the folder `path_to_repo/Datasets/BSDS500/Original`. Then run the following commands :

```
cd path_to_repo/data_retrieval
python3 generate_attacks.py path_to_repo/Datasets/BSDS500/Original path_to_repo/Datasets/BSDS500/Attacks
```

This will create the folder `path_to_repo/Datasets/BSDS500/Attacks` containing attack variations of the original images.

## Kaggle memes dataset

For this dataset, please download the data. After you dezip the file, you will get a folder named `archive`. 

```
cd path_to_repo/data_retrieval
python3 kaggle_splitter.py path_to_archive/reddit_posts.csv path_to_archive/images/kaggle_images path_to_archive/images/Templates
```

This will create a new folder `path_to_archive/images/Templates` containing templates for each class. Then please copy the folder `path_to_archive/images/kaggle_images` under the new name and path `path_to_repo/Datasets/Kaggle_memes/Memes`, and the folder `path_to_archive/images/Templates` to the new path `path_to_repo/Datasets/Kaggle_memes/Templates`.


