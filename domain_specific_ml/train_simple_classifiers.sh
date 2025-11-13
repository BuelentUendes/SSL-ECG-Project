#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
fractions=(0.01 0.025 0.05 0.1 0.25 0.5 1.0)
seeds=(3 5 7 9 42)
models=("logistic_regression")

for model in "${models[@]}"
do
  for seed in "${seeds[@]}"
  do
    for fraction in "${fractions[@]}"
    do
      python3 train_simple_classifiers.py --fs 1000 --window_size 10 --step_size 5 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"
      python3 train_simple_classifiers.py --fs 1000 --window_size 30 --step_size 10 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"
      python3 train_simple_classifiers.py --fs 1000 --window_size 30 --step_size 15 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"
      python3 train_simple_classifiers.py --fs 1000 --window_size 30 --step_size 5 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"

      python3 train_simple_classifiers.py --fs 500 --zero_shot_evaluation --zero_shot_dataset "stressid" --window_size 10 --step_size 5 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"
      python3 train_simple_classifiers.py --fs 700 --zero_shot_evaluation --zero_shot_dataset "wesad" --window_size 10 --step_size 5 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"

      python3 train_simple_classifiers.py --fs 500 --zero_shot_evaluation --zero_shot_dataset "stressid" --window_size 30 --step_size 10 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"
      python3 train_simple_classifiers.py --fs 700 --zero_shot_evaluation --zero_shot_dataset "wesad" --window_size 30 --step_size 10 --classifier_model "$model" --seed "$seed" --label_fraction "$fraction"

    done
  done
done



