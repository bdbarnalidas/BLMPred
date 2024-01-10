#!/bin/bash
#
#SBATCH --job-name=blmpred_5to60
#SBATCH --output=blmpred_5to60.out
#SBATCH --error=blmpred_5to60.err
#SBATCH --nodes=1
#SBATCH --partition=GPU-A40
#SBATCH --nodelist=gpunode09
#SBATCH --ntasks=1
#SBATCH --mem=10G

# Print the task id.
echo "blmpred_5to60"

# Write something useful here...
python3.9 blmpred_5to60.py
