#!/bin/bash
# PREREG_R8 chain, GPU 0, sequential. Log: /data1/users/yangrui/results/idea67b/exp/prl_v2/round7/matrix_r8.log
cd /data1/users/yangrui/projects/idea67b/experiment/code/prl_v2/round7
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true
SPLITS_OVERRIDE="s3 s4" V0=500 bash scripts/run_matrix.sh "0 1 2" m1
SPLITS_OVERRIDE="s3 s4" V0=520 bash scripts/run_matrix.sh "0 1 2" m3p
SPLITS_OVERRIDE="s3 s4" V0=540 bash scripts/run_matrix.sh "0" m5
SPLITS_OVERRIDE="s3 s4" V0=560 bash scripts/run_matrix.sh "0" m4
echo "R8_CHAIN_DONE $(date)"
