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
#SBATCH -t 15:00:00
#SBATCH --export=ALL

module load gpu
module load slurm

. ~/.bashrc
cd ../..
conda activate poyo

eid=$1
behav=$2
config_name=train_ibl_${behav}_${eid}.yaml
ckpt_name=${eid}_${behav}
python eval_single_task_poyo.py --eid $eid \
                                --behavior $behav \
                                --config_name $config_name \
                                --ckpt_name $ckpt_name

conda deactivate
cd scripts/ppwang