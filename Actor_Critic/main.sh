#!/bin/bash
#
#SBATCH -J test
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/Actor_Critic

max_ts=1000
n_epochs=1000

python3 main.py --n_epochs "$n_epochs" --max_steps "$max_ts"