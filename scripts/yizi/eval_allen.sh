#!/bin/bash
#SBATCH --account bcxj-delta-gpu 
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="eval-allen"
#SBATCH --output="eval-allen.%j.out"
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000  
#SBATCH --gpus=1
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc
conda activate poyo

session_id=${1}
behavior=${2}

if [ "$behavior" = "gabors" ]; then
    behave_log_name="gabor_orientation"
else
    behave_log_name=$behavior
fi

config_name=train_allen_${behavior}.yaml
ckpt_name=mouse_${session_id}_${behave_log_name}

cd ../../eval/scripts/allen_visual_behavior_neuropixels/

python eval_single_task_poyo.py --session_id $session_id \
                                --behavior $behavior \
                                --config_name $config_name \
                                --ckpt_name $ckpt_name

conda deactivate

cd scripts/yizi
