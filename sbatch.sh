#!/bin/bash
#SBATCH --job-name=modolo
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=newton3,newton4,newton5,bruno2,bruno4
echo $1
cleanup() {
    echo "byye $1"
}
trap cleanup EXIT

if [ $# -lt 1 ]; then
    echo "Usage: sbatch sbatch.sh <python_script>"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate FSdock

hostname
cd /home/alon.kitin/fs-dock

srun --export=ALL --cpu-bind=none python sbatch_main.py $1

echo $1

# This script performs [brief description of what the script does].
# 
# Usage:
# To run this script using SLURM, use the following command:
# sbatch [script_name.sh]
#
# Ensure that you have the necessary permissions and resources allocated in your SLURM environment.

#  2004  sbatch sbatch.sh /home/alon.kitin/fs-dock/configs/gemini/gemini_cross_no_esm.yaml
#  2005  sbatch sbatch.sh /home/alon.kitin/fs-dock/configs/gemini/gemini_cross_no_groups.yaml 
#  2006  sbatch sbatch.sh /home/alon.kitin/fs-dock/configs/gemini/gemini_cross_no_receptors.yaml 
#  2007  sbatch sbatch.sh /home/alon.kitin/fs-dock/configs/gemini/gemini_cross_no_vae.yaml 