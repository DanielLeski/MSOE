#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --time=60
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=dh-node2
srun singularity shell -B /data:/data -B /scratch:/scratch /data/cs3450/pytorch21q4.4.sif python gan.py
