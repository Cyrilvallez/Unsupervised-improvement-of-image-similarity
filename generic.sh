#!/bin/bash

#SBATCH --job-name=generic
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/cyril.vallez/Project2

# Verify working directory
echo $(pwd)

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate faiss

python3 "$@"

conda deactivate
