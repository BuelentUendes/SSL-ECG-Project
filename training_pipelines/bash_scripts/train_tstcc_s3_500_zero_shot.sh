#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)

for seed in "${seeds[@]}"
do
  # We need the full label fraction here to to transfer
  python3 tstcc_train_cleaned_cv.py --force_retraining --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --use_s3_layers --fs 500
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --use_s3_layers --fs 500 --zero_shot_evaluation --zero_shot_dataset "stressid"
done

