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
#SBATCH -t 0-06:00:00
#SBATCH --export=ALL

. ~/.bashrc

cd ../..

conda activate poyo

eid=$1
train_config=$2

dataset_config=${train_config%.yaml}    # removes .yaml
dataset_config=${dataset_config#train_} # removes train_ prefix
dataset_config="${dataset_config}.yaml"

dataset_path=$(grep "dataset: ibl" "./configs/${train_config}" | sed 's/.*dataset: //')
default_eid=$(echo $dataset_path | sed 's/.*ibl\///;s/\/.*//')

sed -i "s/- dataset: ibl\/${default_eid}\/${dataset_config}/- dataset: ibl\/${eid}\/${dataset_config}/" ./configs/${train_config}

python train.py --config-name $train_config

conda deactivate

cd scripts/yizi
