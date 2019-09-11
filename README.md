# Graph Neural Networks for particle track reconstruction

This repository contains the PyTorch implementation of the GNNs for particle
track reconstruction from CTD 2018: https://arxiv.org/abs/1810.06111.

## Contents

The main python scripts for running:
- *[prepare.py](prepare.py)*: the data preparation script which reads
TrackML data files, cleans and reduces the data, and writes hit graphs to
the filesystem.
- *[heptrx_nnconv.py](heptrx_nnconv.py)*: the main training script
- *[heptrx_nnconv_test.py](heptrx_nnconv_test.py)*: the main testing script

## Instructions

### Setup
```bash
source setup.sh
```

### Prepare files
Note to change input and output data locations. Input data is available here: https://www.kaggle.com/c/trackml-particle-identification/data or on `correlator2.fnal.gov`. This step converts input data into graph data.
```bash
python prepare.py configs/prep_small.yaml
```

### Train an EdgeNet model
```bash
python heptrx_nnconv.py
```

### Evaluate a trained EdgeNet model
```bash
python heptrx_nnconv_test.py [model.pth]
```
