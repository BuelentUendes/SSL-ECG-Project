#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(1.0)
seeds=(7) #3 5 7 9 42 (5 seeds should be enough)
held_out_participants=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

# This is for LP
for held_out_participant in "${held_out_participants[@]}"
do
  for seed in "${seeds[@]}"
  do
    for fraction in "${fractions[@]}"
    do
      python3 tstcc_train_cleaned_cv.py --fs 700 --use_pretrained_encoder --fine_tune_encoder --loso --held_out_participant $held_out_participant --seed "$seed" --label_fraction "$fraction" --window_size 10 --step_size 5
    done
  done
done
