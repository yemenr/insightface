#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=8
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice


#result on split -1, without fbn
#e: 87.0, h: 86.7, d: 86.6, a: 86.7 , c: 86.9


CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvala --prefix ./model/iqiyia1
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvalc --prefix ./model/iqiyic1
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvald --prefix ./model/iqiyid1
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvale --prefix ./model/iqiyie1
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvalh --prefix ./model/iqiyih1

CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvala --prefix ./model/iqiyia2 --fbn
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvalc --prefix ./model/iqiyic2 --fbn
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvald --prefix ./model/iqiyid2 --fbn
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvale --prefix ./model/iqiyie2 --fbn
CUDA_VISIBLE_DEVICES='7' python2 -u train_mlp.py --data /gpu/data2/jiaguo/iqiyi/trainvalh --prefix ./model/iqiyih2 --fbn

python2 -u run.py 11 a 1
python2 -u run.py 11 c 1
python2 -u run.py 11 d 1
python2 -u run.py 11 e 1
python2 -u run.py 11 h 1
python2 -u run.py 11 a 2
python2 -u run.py 11 c 2
python2 -u run.py 11 d 2
python2 -u run.py 11 e 2
python2 -u run.py 11 h 2

