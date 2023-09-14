#!/bin/bash
#
#SBATCH -J interact
#SBATCH -o output.log
#SBATCH -e error.log          
#SBATCH -n 1
#SBATCH --gres=gpu:2

cd ~/dai/SimpleRL/PPO
source ppo/bin/activate

n_epochs=4000000000
n_envs=16
epsilon=0.2
env_name='Walker2d-v4'
n_step_per_batch=1024
vf_coef=0.5
ent_coef=0.01

python3 main.py --n_epochs $n_epochs --n_envs $n_envs --epsilon $epsilon --env_name $env_name --n_step_per_batch $n_step_per_batch --vf_coef $vf_coef --ent_coef $ent_coef
