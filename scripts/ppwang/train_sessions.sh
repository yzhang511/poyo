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
#SBATCH -t 8:00:00
#SBATCH --export=ALL

module load gpu
module load slurm

. ~/.bashrc
cd ..
conda activate poyo

while IFS= read -r line
do
    python create_config.py --eid $line \
                            --base_path ../ \
                            --multitask

    python calculate_normalization_scales.py --data_root /home/ywang74/Dev/poyo_ibl/data/processed \
                                            --dataset_config  /home/ywang74/Dev/poyo_ibl/configs/dataset/ibl_multitask_$line.yaml
done < ../data/train_eids.txt

# unify the normalized config files
python unify_config.py --base_path ../ \
                        --eid_list_path ../data/test_eids.txt 
cd .. 
# echo "Finished calculating normalization scales"
python train.py --config-name train_ibl_sessions.yaml

conda deactivate
cd scripts/ppwang