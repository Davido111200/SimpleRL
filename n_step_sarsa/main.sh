#!/bin/bash
#
#SBATCH -J interact
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/n_step_sarsa

n_epochs=1000
max_ts=1000
n=10

python3 main.py --n_epochs "$n_epochs" --max_ts "$max_ts" --n "$n"