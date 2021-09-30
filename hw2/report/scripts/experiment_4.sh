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

batch=$1
lr=$2

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b ${batch} -lr ${lr} \
  --exp_name q4_b${batch}_r${lr} --n_workers 5

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b ${batch} -lr ${lr} -rtg \
  --exp_name q4_b${batch}_r${lr}_rtg --n_workers 5

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b ${batch} -lr ${lr} --nn_baseline \
  --exp_name q4_b${batch}_r${lr}_nnbaseline --n_workers 5

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b ${batch} -lr ${lr} -rtg --nn_baseline \
  --exp_name q4_b${batch}_r${lr}_rtg_nnbaseline --n_workers 5
