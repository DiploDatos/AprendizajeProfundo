#!/usr/bin/env bash

set -ex

# Be sure the correct nvcc is in the path with the correct pytorch installation
export CUDA_HOME=/opt/cuda/10.1
export PATH=$CUDA_HOME/bin:$PATH

python -m experiment.mlp \
    --train-data ./data/meli-challenge-2019/spanish.train.csv.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt \
    --language spanish
    --validation-data ./data/meli-challenge-2019/spanish.validation.csv.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.3
