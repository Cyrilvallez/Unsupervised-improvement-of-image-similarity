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