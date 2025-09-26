#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)
models=("cnn")

for model in "${models[@]}"
do
  for seed in "${seeds[@]}"
  do
    python3 supervised_training_cleaned_cv.py --dataset "stressid" --fs 500 --model_type "$model" --seed "$seed" --label_fraction 1.0 --batch_size 64 --use_pretrained_encoder --fine_tune_encoder --classifier_head "logistic_regression"
    python3 supervised_training_cleaned_cv.py --dataset "wesad" --fs 700 --model_type "$model" --seed "$seed" --label_fraction 1.0 --batch_size 64 --use_pretrained_encoder --fine_tune_encoder --classifier_head "logistic_regression"
  done
done



