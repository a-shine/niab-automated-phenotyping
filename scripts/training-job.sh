#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source activate niab

srun python scripts/train_plant_seg.py
