#!/bin/sh
 ## Bash script for running several experiments
 seeds=(1 3 5 7 42)
 # shellcheck disable=SC2068
 # shellcheck disable=SC1073
 # shellcheck disable=SC1061

 for i in "${seeds[@]}"
 do
   python3 ./training_pipelines/tstcc_soft_train_cleaned.py --seed $i --force_retraining --gpu 0 --pretrain_all_conditions
   python3 ./training_pipelines/ts2vec_train_cleaned.py --seed $i --force_retraining --gpu 0 --pretrain_all_conditions
   python3 ./training_pipelines/ts2vec_soft_train_cleaned.py --seed $i --force_retraining --gpu 0 --pretrain_all_conditions
 done
