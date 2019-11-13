#!/usr/bin/env bash
#export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export CUDA_VISIBLE_DEVICES='0'

MODEL_DIR="/home/ubun/camel/wp/projects/insightface/recognition/models/20191104153210/r50-marginface-retinaface"

nohup python -u verification_.py --gpu 0 --data-dir /home/ubun/camel/data/retina_school --model "${MODEL_DIR}/models,0" --target lfw,cfp_ff,cfp_fp,agedb_30,surveillance --batch-size 64 > ${MODEL_DIR}/version.txt 2>&1 &
