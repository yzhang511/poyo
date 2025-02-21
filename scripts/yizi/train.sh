#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="train"
#SBATCH --output="train.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000  
#SBATCH --gpus=1
#SBATCH -t 0-04:00:00
#SBATCH --export=ALL

. ~/.bashrc

cd ../..

conda activate poyo

train_config=$1

python train.py --config-name $train_config

conda deactivate

cd scripts/yizi
