#!/bin/bash
#
#SBATCH -J interact
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/PPO

n_epochs=1000000
n_cpu=4
n_trials=3
env_name='Hopper-v4'
epsilon=0.2
batch_size=64

python3 main.py --n_epochs $n_epochs --n_cpu $n_cpu --n_trials $n_trials --env_name $env_name --epsilon $epsilon --batch_size $batch_size


