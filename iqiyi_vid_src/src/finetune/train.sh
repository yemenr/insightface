#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

CUDA_VISIBLE_DEVICES='4,5,6,7' python2 -u fine-tune.py --pretrained-model model/resnet-50 --load-epoch 0 --gpus 0,1,2,3 --model-prefix model/iqiyi-scene2 --batch-size 256 --num-classes 4935 --num-examples 1630000 #2>&1 | tee train448.log

