#!/bin/bash
#SBATCH --job-name=subsampled_unlearning
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=public

# Print GPU info for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi failed - no GPU on this node!"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning-env

# Print environment info
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

# Run the experiment
python real_main.py --config configs/cifar10_subsampled.yaml

# Print completion info
echo "======================================"
echo "End time: $(date)"
echo "======================================"
