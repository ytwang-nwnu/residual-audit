#!/bin/bash
# Python launcher for env deeprehabpile with the pip-installed CUDA libraries on
# LD_LIBRARY_PATH. tensorflow 2.16.1 (pinned by the DeepRehabPile README) does not
# preload the nvidia-*-cu12 pip packages by itself (fixed upstream in TF 2.17), so
# without this it reports "Cannot dlopen some GPU libraries" and runs on CPU.
# Also points XLA at the pip cuda_nvcc package (ptxas), makes a cuDNN autotuning
# failure fall back to a default algorithm instead of aborting the run (the
# fatal "Failed to determine best cudnn convolution algorithm ... RESOURCE_EXHAUSTED"
# seen when several processes share a card), and keeps TF memory growth on.
ENV=/data1/users/yangrui/envs/deeprehabpile
SP=$ENV/lib/python3.11/site-packages
NV=$SP/nvidia
export LD_LIBRARY_PATH=$(echo $NV/*/lib | sed -r 's/\s+/:/g')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$NV/cuda_nvcc --xla_gpu_strict_conv_algorithm_picker=false"
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
exec $ENV/bin/python "$@"
