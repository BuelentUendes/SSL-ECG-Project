#!/bin/sh
 ## Bash script for running several experiments
fractions=(0.01 0.05 0.1 0.25 0.5 1.0)
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 for fraction in "${fractions[@]}"
 do
   python3 supervised_training_cleaned_cv.py --model_type cnn --gpu 0 --label_fraction $fraction --batch_size 64 --force_retraining
   python3 supervised_training_cleaned_cv.py --model_type tcn --gpu 0 --label_fraction $fraction --batch_size 64 --force_retraining
   python3 supervised_training_cleaned_cv.py --model_type transformer --gpu 0 --label_fraction $fraction --batch_size 64 --force_retraining
 done
