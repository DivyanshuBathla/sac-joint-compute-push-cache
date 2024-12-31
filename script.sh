#!/bin/bash

# Define the common arguments
COMMON_ARGS="--env-name MultiTaskCore --policy Gaussian --eval True --gamma 0.99 --tau 0.005 --lr 0.0003 --alpha 0.2 --automatic_entropy_tuning False --seed 123456 --batch_size 256 --num_steps 5000001 --hidden_size 256 --updates_per_step 1 --start_steps 10000 --target_update_interval 1000 --replay_size 1000000"

# List of experiment cases
cases=("case2" "case3" "case4" "case6" "case7")

# Loop through each case and run the script
for case in "${cases[@]}"; do
    echo "Running experiment for case: $case"
    python main.py $COMMON_ARGS --exp-case $case
done
