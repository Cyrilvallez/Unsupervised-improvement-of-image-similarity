#!/bin/bash

#SBATCH --job-name=DBSCAN
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=50000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/cyril.vallez/Project2

# Verify working directory
echo $(pwd)

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate faiss

python3 DBSCAN.py "$@"

conda deactivate
