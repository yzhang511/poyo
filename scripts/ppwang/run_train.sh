#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="poyo"
#SBATCH --output="mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 2-00
#SBATCH --export=ALL
eid=${1}

module load gpu
module load slurm

. ~/.bashrc

conda activate poyo
while IFS= read -r line
do
    cd ..
    python create_config.py --eid ${line} \
                            --base_path ../

    python calculate_normalization_scales.py --data_root /home/ywang74/Dev/poyo_ibl/data/processed \
                                            --dataset_config  /home/ywang74/Dev/poyo_ibl/configs/dataset/ibl_wheel_${line}.yaml 

    python calculate_normalization_scales.py --data_root /home/ywang74/Dev/poyo_ibl/data/processed \
                                            --dataset_config  /home/ywang74/Dev/poyo_ibl/configs/dataset/ibl_whisker_${line}.yaml 

    cd ppwang

    sbatch train.sh "/home/ywang74/Dev/poyo_ibl/configs/train_ibl_wheel_${line}.yaml"
    sbatch train.sh "/home/ywang74/Dev/poyo_ibl/configs/train_ibl_whisker_${line}.yaml"
    sbatch train.sh "/home/ywang74/Dev/poyo_ibl/configs/train_ibl_choice_${line}.yaml"
    sbatch train.sh "/home/ywang74/Dev/poyo_ibl/configs/train_ibl_block_${line}.yaml"
done < /home/ywang74/Dev/poyo_ibl/data/test_eids.txt
