#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)

for seed in "${seeds[@]}"
do
  python3 tstcc_train_cleaned_cv.py --use_pretrained_encoder --fine_tune_encoder --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --fs 700
done

