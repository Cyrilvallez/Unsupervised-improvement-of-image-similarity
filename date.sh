#!/bin/bash

#SBATCH --job-name=main
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/cyril.vallez/Project2

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate faiss

# Exact timestamp before execution
echo Time before execution of script : $(date)

python3 date.py

# Exact timestamp after execution
echo Time after execution of script : $(date)

conda deactivate
