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
    sbatch eval.sh $line choice
    sbatch eval.sh $line block
    sbatch eval.sh $line wheel
    sbatch eval.sh $line whisker
done < /home/ywang74/Dev/poyo_ibl/data/test_eids.txt
