#!/bin/bash

function run_search() {
  for b in 10000 30000 50000
  do
    for r in 0.005 0.01 0.02
    do
      python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
        --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline \
        --exp_name q4_search_b${b}_lr${r}_rtg_nnbaseline
    done
  done
}

# run_search

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $1 -lr $2 \
  --exp_name q4_b$1_r

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $1 -lr $2 -rtg \
  --exp_name q4_b$1_r$2_rtg

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $1 -lr $2 --nn_baseline \
  --exp_name q4_b$1_r$2_nnbaseline

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $1 \
  --exp_name q4_b$1_r$2_rtg_nnbaseline
