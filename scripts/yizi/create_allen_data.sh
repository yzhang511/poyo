#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc

session_id=${1}

user_name=$(whoami)

conda activate poyo

cd ../..
cd data/scripts/allen_visual_behavior_neuropixels

python prepare_data.py --session_id ${session_id} \
       --input_dir /projects/bcxj/$user_name/allen/datasets/raw/ \
       --output_dir /projects/bcxj/$user_name/allen/datasets/processed/

cd ../../../scripts/yizi

conda deactivate
