#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)

for seed in "${seeds[@]}"
do
  python3 simclr_train_cleaned_cv.py --force_retraining --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.1 --use_s3_layers --gpu 1
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.05 --use_s3_layers --gpu 1
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.01 --use_s3_layers --gpu 1
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.025 --use_s3_layers --gpu 1
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.25 --use_s3_layers --gpu 1
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.5 --use_s3_layers --gpu 1
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --use_s3_layers --gpu 1
done

