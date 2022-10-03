#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:2

#SBATCH --output=gpu_job-%j.out
#SBATCH --error=gpu_job-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=10G
#SBATCH --time=1-23:59:00

#Load needed modules
module load gnu10
module load python
pip install terminaltables

#Execute
python train.py --epochs=1
