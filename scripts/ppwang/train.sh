#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="poyo-train"
#SBATCH --output="poyo-train.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 2-00
#SBATCH --export=ALL

module load gpu
module load slurm

. ~/.bashrc
cd ../..
conda activate poyo

train_config=$1

# python train.py --config-name train_ibl_choice.yaml

# python train.py --config-name train_ibl_block.yaml

# python train.py --config-name train_ibl_wheel.yaml

python train.py --config-name $train_config

conda deactivate
cd scripts/ppwang