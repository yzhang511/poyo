#!/bin/bash

# module load gpu
# module load slurm

. ~/.bashrc
cd ../..
conda activate poyo

python eval_result.py

cd scripts/ppwang
conda deactivate