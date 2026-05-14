#!/bin/bash
#SBATCH --job-name=ECG_TS2Vec
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=b.uendes@vu.nl
#SBATCH --output=ecg_ts2vec_%j.out
#SBATCH --error=ecg_ts2vec_%j.err
#SBATCH -C A4000

# Useful bash commands:
## sinfo -N -l
## sinfo -e -o  "%9P %.6D %4X %4Y %36N %32f"

# Test CUDA availability and basic torch functionality
module load cuda12.3/toolkit
module load cuDNN/cuda12.3
source activate ECG-Project

python << EOF
import torch
import os
from datetime import datetime
print(f"Current time: {datetime.now()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")

    # Test basic GPU operations
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✓ GPU matrix multiplication test passed")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("✗ CUDA not available - check GPU allocation and drivers")

print("Environment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS', 'Not set')}")
EOF

echo "=== GPU Test Completed ==="
# The first run runs it and retrains it for the specific seed
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 0.1 --force_retraining --ts2vec_use_gradient_clipping
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 0.01 --ts2vec_use_gradient_clipping
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 0.025 --ts2vec_use_gradient_clipping
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 0.05 --ts2vec_use_gradient_clipping
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 0.25 --ts2vec_use_gradient_clipping
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 0.5 --ts2vec_use_gradient_clipping
python3 ts2vec_train_cleaned_cv.py --seed $1 --ts2vec_epochs 40 --label_fraction 1.0 --ts2vec_use_gradient_clipping

# Command to run the job
#  for SEED in 3; do
#      sbatch --job-name=ECG_TS2Vec_CLIP_seed_${SEED} \
#             --output=ecg_ts2vec_CLIP_${SEED}_%j.out \
#             --time=60:00:00 \
#             --error=ecg_ts2vec_CLIP_${SEED}_%j.err \
#             ./slurm_jobs/slurm_job_ts2vec_clip.sh $SEED
#  done
###