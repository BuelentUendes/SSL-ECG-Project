#!/bin/sh
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 ## Bash script for running several experiments
seeds=(3 5 7 9 42)

for seed in "${seeds[@]}"
  do
    python3 supervised_training_cleaned_cv.py --dataset "ours" --force_retraining --fs 700 --model_type "cnn" --seed "$seed" --label_fraction 1.0 --batch_size 64
      python3 supervised_training_cleaned_cv.py --dataset "ours" --fs 700 --model_type "cnn" --seed "$seed" --label_fraction 1.0 --batch_size 64 --zero_shot_evaluation --zero_shot_dataset "wesad"
  done
done



