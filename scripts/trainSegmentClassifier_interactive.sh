#!/bin/bash

mkdir -p logs
. scripts/setup.sh

export OMP_NUM_THREADS=32
#medium dataset
srun -n 1 python -m torch.utils.bottleneck ./train.py -v -d configs/segclf_med.yaml 2>&1|tee med.out.txt
#small dataset
#srun -n 1 python ./train.py -v -d configs/segclf_small.yaml 2>&1|tee small.out.txt
