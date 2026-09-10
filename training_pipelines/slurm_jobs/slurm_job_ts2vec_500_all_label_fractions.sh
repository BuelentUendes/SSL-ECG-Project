#!/bin/bash
#SBATCH --job-name=ECG_TS2Vec_500
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=b.uendes@vu.nl
#SBATCH -C A4000

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
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 0.1 --fs 500 --force_retraining --ts2vec_masking_prob 0.7
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 0.01 --fs 500 --ts2vec_masking_prob 0.7
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 0.025 --fs 500 --ts2vec_masking_prob 0.7
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 0.05 --fs 500 --ts2vec_masking_prob 0.7
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 0.25 --fs 500 --ts2vec_masking_prob 0.7
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 0.5 --fs 500 --ts2vec_masking_prob 0.7
python3 ts2vec_train_cleaned_cv.py --seed $1 --label_fraction 1.0 --fs 500 --ts2vec_masking_prob 0.7

# Command to run the job
#  for SEED in 200 201 202 203 204 205 206 207 208 209; do
#      sbatch --job-name=ECG_500_TS2VEC_seed_${SEED} \
#             --output=ecg_500_ts2vec_${SEED}_%j.out \
#             --error=ecg_500_ts2vec_${SEED}_%j.err \
#             --time=48:00:00 \
#             ./slurm_jobs/slurm_job_ts2vec_500_all_label_fractions.sh $SEED
#  done
##