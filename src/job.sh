#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=SATproject
#SBATCH --output=output/out.txt
#SBATCH --error=output/err.txt

make build

srun sat_to_matrix_mult