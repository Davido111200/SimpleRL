#!/bin/bash


# Set the values for n_epochs
n_runs=100
n_epochs=400


python3 main.py --n_epochs "$n_epochs" --n_runs "$n_runs"