#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(0.01 0.05 0.1 0.25 0.5 1.0)
seeds=(3 5 7 9)

for seed in "${seeds[@]}"
do
  python3 tstcc_train_cleaned_cv.py --force_retraining --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.1
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.05
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.01
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.025
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.25
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 0.5
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0
done

