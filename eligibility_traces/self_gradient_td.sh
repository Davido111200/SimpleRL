#!/bin/bash


# Set the values for n_epochs and max_ts
n_epochs=2
max_ts=1000

python3 self_gradient_td.py --n_epochs "$n_epochs" --max_ts "$max_ts"