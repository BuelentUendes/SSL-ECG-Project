#!/bin/bash
#SBATCH --job-name=ECG_SimCLR
#SBATCH --time=0:10:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=b.uendes@vu.nl
#SBATCH --output=ecg_%j.out
#SBATCH --error=ecg_%j.err

module add cuda12.3/toolkit/12.3
source activate /var/scratch/bun201/SSL-ECG-Project/ECG-Project

python3 simclr_train_cleaned_cv.py --seed 12345 --epochs 2
