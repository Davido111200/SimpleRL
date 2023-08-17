#!/bin/bash
#
#SBATCH -J interact
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/Actor_Critic

max_ts=3000
n_epochs=10000
n_trials=5

python3 main.py --n_epochs "$n_epochs" --max_steps "$max_ts" --n_trials "$n_trials"