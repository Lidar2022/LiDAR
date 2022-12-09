#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:4

#SBATCH --output=gpu_job-%j.out
#SBATCH --error=gpu_job-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks=32

#SBATCH --mem=256G
#SBATCH --time=4-23:59:00

set echo 
umask 0022 
# to see ID and state of GPUs assigned 
nvidia-smi

#Load needed modules
module load gnu10
module load python
module load cudnn
module load nccl
#Execute
python train.py --dist-url 'tcp://127.0.0.1:29500' --resume_path ../checkpoints/complexer_yolo/Model_complexer_yolo_epoch_4280.pth --start_epoch 4281 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --cfgfile 'config/cfg/complex_yolov3.cfg' --batch_size 32 --tensorboard_freq 25 --num_epochs 8000 
