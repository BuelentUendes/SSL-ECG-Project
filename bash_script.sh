#!/bin/sh
 ## Bash script for running several experiments
 seeds=(1 3 5 7)
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 for i in "${seeds[@]}"
 do
   python3 ./training_pipelines/tstcc_train_cleaned.py --seed $i --force_retraining --gpu 0
   python3 ./training_pipelines/tstcc_train_cleaned.py --seed $i --force_retraining --gpu 0 --pretrain_all_conditions
 done
