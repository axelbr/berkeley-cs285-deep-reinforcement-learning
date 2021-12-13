#!/bin/bash

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --use_rnd --use_simhash --simhash_k 128 --exp_name q1_simhash_med  & \
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --use_rnd --use_simhash --simhash_k 128 --exp_name q1_simhash_hard