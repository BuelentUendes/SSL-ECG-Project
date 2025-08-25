#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(0.01 0.05 0.1 0.25 0.5 1.0)
seeds=(3 5 7 9)
models=("cnn" "tcn" "transformer")

for model in "${models[@]}"
do
  for seed in "${seeds[@]}"
  do
    for fraction in "${fractions[@]}"
    do
      python3 supervised_training_cleaned_cv.py --dataset "ours" --scoring_metric "roc_auc" --fs 1000 --model_type "$model" --seed "$seed" --label_fraction "$fraction" --batch_size 64 --gpu 0 --force_retraining
    done
  done
done



