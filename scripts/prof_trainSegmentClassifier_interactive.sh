#!/bin/bash

mkdir -p logs
. scripts/setup.sh

export OMP_NUM_THREADS=32
#medium dataset
srun -n 1 python -m cProfile -o ${SLURMD_NODENAME}.prof.med.out.txt ./train.py -v -d configs/segclf_med_ep1.yaml 2>&1|tee ${SLURMD_NODENAME}.med.out.txt

#vtune
#srun -n 1 amplxe-cl -collect hotspots -r vtune-profs -trace-mpi python -m cProfile -o ${SLURMD_NODENAME}.prof.ep1.med.out.txt ./train.py -v -d configs/segclf_med_ep1.yaml 2>&1|tee ${SLURMD_NODENAME}.med.out.txt

#ep=16
#srun -n 1 python -m cProfile -o ${SLURMD_NODENAME}.prof.ep${ep}.med.out.txt ./train.py -v -d configs/segclf_med_ep${ep}.yaml 2>&1|tee ${SLURMD_NODENAME}.med.${ep}.out.txt

#small dataset
#srun -n 1 python ./train.py -v -d configs/segclf_small.yaml 2>&1|tee small.out.txt
