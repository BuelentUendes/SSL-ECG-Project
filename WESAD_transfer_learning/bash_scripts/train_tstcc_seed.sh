#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(1.0)
seeds=(5 7 9 11 13 15 17 19 42) #Seed 3 was already run

for seed in "${seeds[@]}"
do
  python3 tstcc_train_cleaned_cv.py --train_ratio_encoder 1.0 --seed "$seed" --label_fraction 1.0 --fs 700
done

