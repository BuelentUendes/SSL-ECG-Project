#!/bin/sh
 ## Bash script for running several experiments
fractions=(0.1 0.25 0.5 1.0)
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 for fraction in "${fractions[@]}"
 do
   python3 supervised_training_cleaned_cv.py --dataset "stressid" --scoring_metric "f1" --fs 500 --model_type cnn --label_fraction $fraction --batch_size 64 --gpu 0 --force_retraining
 done


