#!/bin/bash
#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="ibl-data"
#SBATCH --output="ibl-data.%j.out"
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc
eid=${1}
unaligned=${2}
user_name=$(whoami)

if [ "$unaligned" = "True" ]; then
    unaligned="--unaligned"
else
    unaligned=""
fi

conda activate poyo

cd ../..
cd data/scripts/ibl_visual_behavior_neuropixels

python prepare_data.py --eid ${eid} \
       --base_path /projects/bcxj/$user_name/ibl/datasets/ \
       $unaligned

cd ../../../scripts/yizi

conda deactivate