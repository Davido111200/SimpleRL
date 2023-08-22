#!/bin/bash
#
#SBATCH -J interact
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/Qlearning

n_epochs=10000
n_trials=50

python3 main.py --n_epochs "$n_epochs" --n_trials "$n_trials"