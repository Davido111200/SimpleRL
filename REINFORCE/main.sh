#!/bin/bash
#
#SBATCH -J interact
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/REINFORCE

n_epochs=1000
max_ts=1000
n_trials=5

python3 main.py --n_epochs "$n_epochs" --max_ts "$max_ts" --n_trials "$n_trials"