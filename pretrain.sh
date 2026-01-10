#!/bin/bash
#SBATCH --job-name=GeCo
#SBATCH --output=train/GeCo_pretrain_%j.txt
#SBATCH --error=train/GeCo_pretrain_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=50197
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

module load Anaconda3
source activate cotr
conda activate base
conda activate cotr

srun --unbuffered python pretrain.py \
--model_name GeCo_PRETRAIN \
--data_path /d/hpc/projects/FRI/pelhanj/fsc147 \
--epochs 150 \
--lr 1e-4 \
--backbone_lr 0 \
--lr_drop 150 \
--weight_decay 1e-4 \
--batch_size 4 \
--tiling_p 0.2