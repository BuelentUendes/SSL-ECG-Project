#!/bin/sh
 ## Bash script for running several experiments
fractions=(0.01 0.05 0.1 0.25 0.5 1.0)
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 for fraction in "${fractions[@]}"
 do
   python3 train_simple_classifiers.py --classifier_model mlp --gpu 0 --label_fraction $fraction
 done
