#!/bin/bash
#SBATCH --job-name=molecules        # Job name
#SBATCH --output=/home/alon.kitin/fs-dock/configs/outputs/%j_output.log           # Output log file (%j = job ID)
#SBATCH --error=/home/alon.kitin/fs-dock/configs/outputs/%j_error.log             # Error log file
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB


echo "$@"
# Run your Python executable
srun python sbatch_main.py "$@"
