#!/bin/bash
#
#SBATCH --job-name=blmpred_8to25
#SBATCH --output=blmpred_8to25.out
#SBATCH --error=blmpred_8to25.err
#SBATCH --nodes=1
#SBATCH --partition=GPU-A40
#SBATCH --nodelist=gpunode10
#SBATCH --ntasks=1
#SBATCH --mem=10G

# Print the task id.
echo "blmpred_8to25"

# Write something useful here...
python3.9 blmpred_8to25.py
