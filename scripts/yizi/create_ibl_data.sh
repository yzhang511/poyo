#!/bin/bash

#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
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
unaligned=${2}
user_name=$(whoami)

conda activate poyo

cd ../..
cd data/scripts/ibl_visual_behavior_neuropixels

python prepare_data.py --eid ${eid} \
       --base_path /projects/bcxj/$user_name/ibl/datasets/ \
       --unaligned ${unaligned}

cd ../../../scripts/yizi

conda deactivate