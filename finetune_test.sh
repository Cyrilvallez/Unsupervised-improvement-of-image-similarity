#!/bin/bash

#SBATCH --job-name=finetune_test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --chdir=/cluster/raid/home/cyril.vallez/Project2

# Verify working directory
echo $(pwd)

# Pull last modifications
git pull

# Print gpu configuration for this job
nvidia-smi

# Verify gpu allocation (should be 1 GPU)
echo "Indices of visible GPU(s) before job : $CUDA_VISIBLE_DEVICES"

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate faiss

python3 finetune.py "$@"

conda deactivate
