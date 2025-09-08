#!/bin/bash
#SBATCH --job-name=ECG             # Job name
#SBATCH --time=24:00:00            # Max run time (HH:MM:SS)
#SBATCH -N 1                       # Number of nodes
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=24         # CPU cores per task
#SBATCH --output=ecg_train.out     # Standard output log
#SBATCH --error=ecg_train.err      # Standard error log

# --- Load CUDA module ---
module add cuda12.3/toolkit/12.3

# --- Set CUDA device visibility (necessary for GPU selection) ---
export CUDA_VISIBLE_DEVICES=0

# --- Initialize Conda ---
source activate /var/scratch/cla224/ECG_env

# --- Set MLflow tracking server ---
export MLFLOW_TRACKING_URI=http://fs0.das6.cs.vu.nl:5005

# --- Run Fully Supervised Model script ---
# CNN
# echo "Launching training with seed: $1"
# python supervised_training.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --model_type "cnn" \
#   --seed $1 \
#   --batch_size 16 \
#   --scheduler_factor 0.5 \
#   --scheduler_min_lr 1e-09 \
#   --patience 20 \
#   --lr 1e-5 \
#   --label_fraction 0.01 \

# TCN
# echo "Launching training with seed: $1"
# python supervised_training.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --model_type "tcn" \
#   --seed $1 \
#   --lr 0.0005 \
#   --patience 20 \
#   --label_fraction 0.01 \

# Transformer
# echo "Launching training with seed: $1"
# python supervised_training.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --model_type "transformer" \
#   --seed $1 \
#   --lr 1e-5 \
#   --patience 20 \
#   --label_fraction 0.5 \

# --- Run Self Supervised Model script ---
# TS2Vec
# echo "Launching training with seed: $1"
# python ts2vec_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --ts2vec_epochs 5 \
#   --label_fraction 0.5 \
#   --seed $1

# TS2VecSoft
# echo "Launching training with seed: $1"
# python ts2vec_soft_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --ts2vec_epochs 5 \
#   --label_fraction 0.01 \
#   --seed $1

# TSTCC
# echo "Launching training with seed: $1"
python tstcc_train.py run \
  --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
  --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
  --tcc_epochs 40 \
  --label_fraction 0.1 \
  --pretrain_all_conditions True \
  --seed $1

# TSTCCSoft
# echo "Launching training with seed: $1"
# python tstcc_soft_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --tcc_epochs 40 \
#   --label_fraction 0.01 \
#   --seed $1

# SimCLR
# echo "Launching training with seed: $1"
# python simclr_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --epochs 300 \
#   --label_fraction 0.01 \
#   --seed $1


# --- Run Transfer Learning script ---
# Transfer learning
# python ppg_ts2vecsoft_train.py run \
#   --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
#   --ecg_window_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
#   --ppg_window_path "../../../../var/scratch/cla224/ECG-Project/data/ppg_windows.h5" \