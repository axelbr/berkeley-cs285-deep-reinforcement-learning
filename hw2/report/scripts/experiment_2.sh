#!/bin/bash

python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $1 -lr $2 -rtg \
--exp_name q2_b$1_r$2