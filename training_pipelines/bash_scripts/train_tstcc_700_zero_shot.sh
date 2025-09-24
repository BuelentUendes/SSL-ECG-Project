#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)

for seed in "${seeds[@]}"
do
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --fs 700 --zero_shot_evaluation --zero_shot_dataset "wesad"
done

