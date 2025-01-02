#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="create-dataset"
#SBATCH --output="create-dataset.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-05
#SBATCH --export=ALL

. ~/.bashrc
eid=${1}
conda activate poyo

cd ../..
cd data/scripts/ibl_repro_ephys

python prepare_data.py --eid ${eid}

cd ../../../scripts/ppwang

conda deactivate