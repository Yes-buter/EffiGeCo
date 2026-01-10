#!/bin/bash
#SBATCH --job-name=GeCo
#SBATCH --output=evaluation/test_GeCo%j.txt
#SBATCH --error=evaluation/test_GeCo%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=50197
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

module load Anaconda3
source activate geco
conda activate base
conda activate geco

srun --unbuffered python evaluate.py \
--model_name GeCo \
--data_path /d/hpc/projects/FRI/pelhanj/fsc147 \
--model_path /d/hpc/projects/FRI/pelhanj/fsc147/models/