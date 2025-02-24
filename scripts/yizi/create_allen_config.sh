#!/bin/bash
#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="allen-config"
#SBATCH --output="allen-config.%j.out"
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc
session_id=${1}
user_name=yzhang39

conda activate poyo

base_path="/u/yzhang39/poyo/"
config_path="configs/dataset/allen"

template_session_id="715093703"
template_configs=("allen_running_speed.yaml" "allen_gabors.yaml" "allen_static_gratings.yaml")

cd ${base_path}/${config_path}

if [ ! -d "${session_id}" ]; then
    mkdir ${session_id}
fi

for template_config in "${template_configs[@]}"; do
    cp -r ${template_session_id}/${template_config} ${session_id}/${template_config}

    sed -i "s/mouse_${template_session_id}/mouse_${session_id}/g" ${session_id}/${template_config}

    cd ${base_path}/scripts

    if [[ "$template_config" != "allen_gabors.yaml" && "$template_config" != "allen_static_gratings.yaml" ]]; then
        python calculate_normalization_scales.py \
            --data_root /projects/bcxj/$user_name/allen/datasets/processed/ \
            --dataset_config ${base_path}/${config_path}/${session_id}/${template_config}
    fi
    cd ${base_path}/${config_path}
done

cd ${base_path}/scripts/yizi

conda deactivate

