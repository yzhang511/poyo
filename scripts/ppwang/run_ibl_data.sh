#!/bin/bash

while IFS= read -r line
do
    sbatch create_ibl_data.sh $line
done < ../../data/train_eids.txt