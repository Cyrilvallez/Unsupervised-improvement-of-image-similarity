# Unsupervised improvement of image similarity

This library provides solutions for extracting features from images, perform experiments on search time in large databases of such features, clustering experiments, and learning augmentation (fine-tuning) of SimCLR trained models. It is organized in 4 main packages :

1. *extractor* : package to extract the features from images using different techniques
2. *fast_search* : package to perform experiments on efficient nearest neighbors search in large databases
3. *clustering* : package to perform experiemnts on clustering of images
4. *finetuning* : package to finetune SimCLR models

There are also other packages being used more or less as helpers for running the main packages :

5. *helpers* : helpers functions
6. *data_retrieval* : processing of the datasets 
7. *SimCLRv1* : Model definitions and downloader for the SimCLRv1 pre-trained models
7. *SimCLRv2* : Model definitions and downloader for the SimCLRv2 pre-trained models

# Usage

In the root of the repository, there are different scripts made for specific tasks. 

### **extract_features.py**

This is designed to extract features from images and save them under the folder `Features` in the repository. This way they can be easily accessed for downstream tasks. You need to specify a model name (or a Pytorch module), the transforms you wish to apply as pre-processing on images, and the datasets of images you want to extract the features from.

You need to manually change these parameters in the script before running it.

### **search_experiment.py**

This is the script used to perform experiments. You need to provide the name of the algorithm from which the features were extracted, the datasets on which you want to perform the experiment, and some parameters depending on the experiment function you choose.

You need to manually change these parameters in the script before running it. Once you changed the parameters, you can run the file as

```
python3 search_experiment.py experiment_name
```

and it will save the results under `Results/experiment_name/results.json

### **process.py**

Process the results from an experiment as performed by `search_experiment.py`. You will need to provide the experiment_name you provided while performing the experiment. Then you can choose what plot to create or not. It will save them in the experiment_name folder.

### **explore_neighbors.py**

The script will save the 10 nearest neighbors of each image in the query_dataset for visual inspection of the quality of retrieval in the database. You need to provide the algorithm name, the datasets names, the metric for image distance, and the folder to which save the results.

You need to manually change these parameters in the script before running it. 

### **hierarchical.py**

This script allows to run hierarchical clustering on a dataset of images. The syntax to run it is the following :

```
python3 hierarchical.py [--algo ALGO [ALGO ...]] [--metric {euclidean,cosine}] \
    [--linkage {single,complete,average,centroid,ward}] [--partition {full,clean}] \
    [--save {True,False}]
```

It will save the cluster assignments and (if --save is True), different plots and cluster attribute in a directory tree starting at `Clustering_results`. To get details about each argument, use 

```
python3 hierarchical.py -h
```

### **DBSCAN.py** 

Run DBSCAN clustering algorithm on a dataset of images. The syntax to run it is the following :

```
python3 DBSCAN.py [--algo ALGO [ALGO ...]] [--metric {euclidean,cosine}] \
   [--precisions PRECISIONS [PRECISIONS ...]] [--samples SAMPLES] \
   [--partition {full,clean}] [--save {True,False}]
```

It will save the cluster assignments and (if --save is True), different plots and cluster attribute in a directory tree starting at `Clustering_results`. To get details about each argument, use 

```
python3 DBSCAN.py -h
```

### **finetune.py**

This will perform augmentative learning (finetuning) for the SimCLR models. The call signature is 

```
python3 finetune.py --train_dataset TRAIN_DATASET --log_dir LOG_DIR \
   [--model MODEL] [--arch_depth ARCH_DEPTH] [--arch_width ARCH_WIDTH] \
   [--arch_sk ARCH_SK] [--val_dataset VAL_DATASET] [--size SIZE] \
   [--jitter JITTER] [--optimizer {lars,adam}] [--lr LR] [--epochs EPOCHS] \
   [--batch_size BATCH_SIZE] [--temperature TEMPERATURE] \
   [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM] \
   [--nesterov {False,True}] [--scheduler {False,True}] [--gpus GPUS] \
   [--workers WORKERS] 
```

It will perform the training and log some metric to `log_dir`, and the model at each epoch under `log_dir_models`. To get details about each argument, use 

```
python3 finetune.py -h
```

### **evaluate_finetuning.py**

This is a quick file to evaluate the improvement/degradation of models during the finetuning process. You need to specify the paths to the model checkpoints during finetuning, and the folder where the figures will be saved.

You need to manually change these parameters in the script before running it.

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

```
python3 download.py -h
```

# Datasets

We personally used 4 datasets that can be found online. They are the [BSDS500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), [Kaggle memes dataset](https://www.kaggle.com/datasets/gmorinan/most-viewed-memes-templates-of-2018), the first half of the [Flickr 1M dataset](https://press.liacs.nl/mirflickr/mirdownload.html) and the [Cartoon classification dataset](https://www.kaggle.com/datasets/volkandl/cartoon-classification). 

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

## Flickr dataset

For the Flickr dataset, please download only the first half (images0.zip to images4.zip on their website), and then put every subfolder of those files (named `0` to `49`) inside a single folder `path_to_repo/Datasets/Flickr500K`.

## Cartoons dataset

This dataset is only used for finetuning the models, and hence does not need to be placed in any specific location in the repository. However if you want to train on it, you will need to provide a valid path to it at runtime.

# Computational setup

For most of the scripts, at least 1 GPU is almost essential. For the finetuning, we do not even give the possibility to run on CPU, as it is absolutely impossible to get results in a realistic time. Moreover, for the finetuning GPUs with a lot of memory(such as a100 GPUs) are very much welcome as it benefits from large batch sizes.