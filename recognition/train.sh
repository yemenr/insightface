#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# arcface LResNet100E-IR
#CUDA_VISIBLE_DEVICES='0' nohup python -u train_seqface.py --network r100 --loss arcface --dataset multidata --auxloss git --per-batch-size 420 > a.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' python -u train_seqface.py --network r100 --loss arcface --dataset multidata --auxloss git --per-batch-size 128
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r100 --loss svxface --dataset emore --per-batch-size 440 > c.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r100 --loss arcface --dataset emore_glint --per-batch-size 420 > a.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r50 --loss svxface --dataset emore_glint --per-batch-size 480 > 128_svxface_emore_glint_20190626.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' python -u train.py --network r50 --loss svxface --dataset emore_glint --per-batch-size 480
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r100 --loss svxface --dataset emore_glint --per-batch-size 400 > a.out 2>&1 &

CUDA_VISIBLE_DEVICES='0' python -u train.py --network r50 --loss svxface --dataset meshface --per-batch-size 480
# triplet LResNet100E-IR
#CUDA_VISIBLE_DEVICES='1' python -u train.py --network r100 --loss triplet --lr 0.005 --dataset emore

# parall train
#CUDA_VISIBLE_DEVICES='0' python -u train_parall.py --network r100 --loss arcface --dataset emore
