#!/bin/bash

env_1=PointmassEasy-v0
env_2=PointmassHard-v0

python cs285/scripts/run_hw5_expl.py --env_name $env_1 --use_rnd --unsupervised_exploration --exp_name q1_${env_2}_rnd & \
python cs285/scripts/run_hw5_expl.py --env_name $env_1 --unsupervised_exploration --exp_name q1_${env_1}_random & \
python cs285/scripts/run_hw5_expl.py --env_name $env_2 --use_rnd --unsupervised_exploration --exp_name q1_${env_2}_rnd & \
python cs285/scripts/run_hw5_expl.py --env_name $env_2 --unsupervised_exploration --exp_name q1_${env_2}_random