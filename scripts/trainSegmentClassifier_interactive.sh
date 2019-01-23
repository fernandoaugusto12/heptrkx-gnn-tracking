#!/bin/bash

mkdir -p logs
. scripts/setup.sh

export OMP_NUM_THREADS=32
srun -n 1 python ./train.py -v -d configs/segclf_small.yaml 2>&1|tee small.out.txt
