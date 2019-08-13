#!/bin/bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
#export CPATH=/usr/local/cuda/include:$CPATH
echo export PATH="$HOME/miniconda/bin:$PATH" >> ~/.bashrc
#echo export CPATH=/usr/local/cuda/include:$CPATH >> ~/.bashrc

#conda create -n torch -c pytorch Python=3.6 numpy cuda100 magma-cuda100 pytorch
conda create -n torch -c pytorch Python=3.6 numpy pytorch torchvision cpuonly

conda init bash

source $HOME/.bashrc

conda activate torch

#get the heptrkx working area, CPU only for now :-(
git clone git@github.com:jmduarte/heptrkx-gnn-tracking.git -b lgray-sparse-hacking
cd heptrkx-gnn-tracking
#get torch geometric
for a in pytorch_scatter pytorch_sparse pytorch_cluster pytorch_spline_conv pytorch_geometric; do 
    git clone https://github.com/rusty1s/$a.git
    pushd $a 
    python setup.py install
    popd 
done

conda install pyyaml tqdm requests matplotlib 
conda install awkward -c conda-forge

