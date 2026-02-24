#!/bin/bash
# Git commit and submit job script

echo "======================================"
echo "Step 1: Checking git status..."
echo "======================================"
git status

echo ""
echo "======================================"
echo "Step 2: Creating new branch..."
echo "======================================"
git checkout -b subsampled-hessian-unlearning

echo ""
echo "======================================"
echo "Step 3: Adding all changes..."
echo "======================================"
git add .

echo ""
echo "======================================"
echo "Step 4: Showing what will be committed..."
echo "======================================"
git status

echo ""
echo "======================================"
echo "Step 5: Committing changes..."
echo "======================================"
git commit -m "Implement Subsampled-Hessian Unlearning Algorithm

- Added subsampled_hessian_unlearning() function in src/forget.py
- Supports both analytical Hessian (linear=true) and conjugate gradient (linear=false)
- Created D_ss subset (30% of training data) for efficient Hessian computation
- Integrated wandb logging for training/validation metrics tracking
- Added MLP flatten layer and model-aware transforms (MLP vs ResNet18)
- Created configs/cifar10_subsampled.yaml with all algorithm parameters
- Fixed partition name in SLURM script (gpu -> public)
- Updated requirements.txt and environment.yml with wandb and scikit-learn
- Commented out old surrogate dataset approach (preserved for reference)

Fixes:
- Memory error from 11TB Hessian matrix (now uses 4GB analytical Hessian for linear mode)
- Shape mismatch error (added nn.Flatten() to MLP model)
- Progressive wandb logging (train_loss and val_acc per epoch)

Ready to run certified unlearning experiments on CIFAR-10!"

echo ""
echo "======================================"
echo "Step 6: Pushing to remote..."
echo "======================================"
git push -u origin subsampled-hessian-unlearning

echo ""
echo "======================================"
echo "Step 7: Submitting SLURM job..."
echo "======================================"
sbatch run_unlearning.sbatch

echo ""
echo "======================================"
echo "Step 8: Checking job queue..."
echo "======================================"
sleep 2
squeue -u $USER

echo ""
echo "======================================"
echo "All done!"
echo "Monitor job with: squeue -u \$USER"
echo "Check logs in: logs/slurm_*.out"
echo "View wandb at: https://wandb.ai"
echo "======================================"
