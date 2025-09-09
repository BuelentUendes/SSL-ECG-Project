#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)

for seed in "${seeds[@]}"
do
  python3 simclr_train_cleaned_cv.py --force_retraining --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.1 --gpu 0 --use_tstcc_encoder
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.05 --gpu 0 --use_tstcc_encoder
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.01 --gpu 0 --use_tstcc_encoder
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.025 --gpu 0 --use_tstcc_encoder
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.25 --gpu 0 --use_tstcc_encoder
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.5 --gpu 0 --use_tstcc_encoder
  python3 simclr_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --gpu 0 --use_tstcc_encoder
done

