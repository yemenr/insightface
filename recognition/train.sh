#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# arcface LResNet100E-IR
#CUDA_VISIBLE_DEVICES='1' python -u train.py --network r100 --loss arcface --dataset emore

# triplet LResNet100E-IR
CUDA_VISIBLE_DEVICES='1' python -u train.py --network r100 --loss triplet --lr 0.005 --dataset emore

# parall train
#CUDA_VISIBLE_DEVICES='1' python -u train_parall.py --network r100 --loss arcface --dataset emore
