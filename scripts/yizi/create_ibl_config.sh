#!/bin/bash
#SBATCH --account=bcxj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="ibl-config"
#SBATCH --output="ibl-config.%j.out"
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

. ~/.bashrc
eid=${1}
unaligned=${2}
user_name=yzhang39

if [ "$unaligned" = "True" ]; then
    unaligned="unaligned"
else
    unaligned="aligned"
fi

conda activate poyo

base_path="/u/yzhang39/poyo/"
config_path="configs/dataset/ibl"

template_eid="d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
template_configs=("ibl_wheel_${unaligned}.yaml" "ibl_whisker_${unaligned}.yaml")

if [ "$unaligned" = "False" ]; then
    template_configs+=("ibl_choice.yaml", "ibl_block.yaml")
fi

cd ${base_path}/${config_path}

if [ ! -d "${eid}" ]; then
    mkdir ${eid}
fi

for template_config in "${template_configs[@]}"; do
    cp -r ${template_eid}/${template_config} ${eid}/${template_config}

    sed -i "s/${template_eid}_${unaligned}/${eid}_${unaligned}/g" ${eid}/${template_config}
    sed -i "s/${template_eid}/${eid}/g" ${eid}/${template_config}

    cd ${base_path}/scripts

    if [[ "$template_config" != "ibl_choice.yaml" && "$template_config" != "ibl_block.yaml" ]]; then
        python calculate_normalization_scales.py \
            --data_root /projects/bcxj/$user_name/ibl/datasets/processed/ \
            --dataset_config ${base_path}/${config_path}/${eid}/${template_config}
    fi
    cd ${base_path}/${config_path}
done

cd ${base_path}/scripts/yizi

conda deactivate

