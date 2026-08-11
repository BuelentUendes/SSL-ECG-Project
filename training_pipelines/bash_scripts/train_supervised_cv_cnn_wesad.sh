#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(1.0)
seeds=(3 5 7 9 11 13 15 17 19 42)
models=("cnn")

for model in "${models[@]}"
do
  for seed in "${seeds[@]}"
  do
    for fraction in "${fractions[@]}"
    do
      python3 supervised_training_cleaned_cv.py --dataset "wesad" --fs 700 --model_type "$model" --seed "$seed" --label_fraction "$fraction" --gpu 0 --force_retraining
    done
  done
done



