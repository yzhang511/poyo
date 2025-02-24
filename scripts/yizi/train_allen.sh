#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="train-allen"
#SBATCH --output="train-allen.%j.out"
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

session_id=$1
train_config=$2

dataset_config=${train_config%.yaml}    # removes .yaml
dataset_config=${dataset_config#train_} # removes train_ prefix
dataset_config="${dataset_config}.yaml"

dataset_path=$(grep "dataset: allen" "./configs/${train_config}" | sed 's/.*dataset: //')
default_session_id=$(echo $dataset_path | sed 's/.*allen\///;s/\/.*//')

sed -i "s/- dataset: allen\/${default_session_id}\/${dataset_config}/- dataset: allen\/${session_id}\/${dataset_config}/" ./configs/${train_config}

python train.py --config-name $train_config

conda deactivate

cd scripts/yizi
