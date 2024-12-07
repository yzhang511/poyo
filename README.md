# Installation
## Environment setup with `venv`
Clone the project, enter the project's root directory, and then run the following:

```bash
conda create -n poyo python=3.9      # create an empty virtual environment
conda activate poyo                  # activate it
pip install -e .                     # install project-kirby into your path
pip install -r ibl_requirements.txt  # install packages needed to load IBL data
```

Currently this project requires the following:
- Python 3.9 (also requires python3.9-dev)
- PyTorch 2.0.0
- CUDA 11.3 - 11.7 
- xformers is optional, but recommended for training with memory efficient attention

## Documentation
> [!WARNING]  
> The docs are hosted publically for convenience, please do not share the link.

You can find the documentation for this project [here](https://chic-dragon-bc9a04.netlify.app/).

## To Run POYO on IBL Data:

1. Prepare datasets:
```
cd data/scripts/ibl_repro_ephys
python prepare_data.py --eid XXX
```

2. Change session ID in data config: `configs/dataset/ibl_choice.yaml` for each behavior

3. Calculate normalization mean and std for continuous behavior:
```
cd scripts/
python calculate_normalization_scales.py --data_root ~/poyo_ibl/data/processed/ --dataset_config ~/poyo_ibl/configs/dataset/ibl_wheel.yaml
```
Paste the mean and std to data config files. 

4. Change the params for unit drop out in `configs/train_ibl_choice.yaml` for each behavior: 
```
- _target_: kirby.transforms.UnitDropout
    max_units: 1000
    min_units: 40
    mode_units: 80
    tail_right: 120
    peak: 10
    M: 10
```
You mainly need to change `min_units` and `mode_units`. 

5. Train (modify this file):
```
sbatch train.sh
```

6. Eval:
```
python eval_single_task_poyo.py --eid XXX --behavior choice --config_name train_ibl_choice.yaml --ckpt_name XXXXX
```
