#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(1.0)
seeds=(3 5 7 9 42) #3 5 7 9 42
models=("logistic_regression")

for model in "${models[@]}"
do
  for seed in "${seeds[@]}"
  do
    for fraction in "${fractions[@]}"
    do
      python3 tstcc_train_cleaned_cv.py  --fs 500 --use_pretrained_encoder --fine_tune_encoder --seed "$seed" --label_fraction "$fraction" --window_size 10 --step_size 5
      python3 tstcc_train_cleaned_cv.py  --fs 500 --use_pretrained_encoder --seed "$seed" --label_fraction "$fraction" --window_size 10 --step_size 5
      # Now the S3 layers
      python3 tstcc_train_cleaned_cv.py  --fs 500 --use_s3_layers --use_pretrained_encoder --fine_tune_encoder --seed "$seed" --label_fraction "$fraction" --window_size 10 --step_size 5
      python3 tstcc_train_cleaned_cv.py  --fs 500 --use_s3_layers --use_pretrained_encoder --seed "$seed" --label_fraction "$fraction" --window_size 10 --step_size 5
    done
  done
done



