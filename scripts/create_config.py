import yaml
import os
import argparse

parser = argparse.ArgumentParser(description='Create a config file')
parser.add_argument('--base_path', type=str, help='Base path for the dataset')
parser.add_argument('--eid', type=str, help='Experiment ID')

args = parser.parse_args()
base_path = args.base_path
base_path = os.path.join(base_path, 'configs')
eid = args.eid
# dataset configs
# dataset choice config
data_choice_path = os.path.join(base_path, 'dataset', 'ibl_choice.yaml')
with open(data_choice_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)

data[0]['selection'][0]['dandiset'] = f'ibl_{eid}'
data[0]['selection'][0]['sortsets'][0] = f'{eid}'

with open(os.path.join(base_path, 'dataset', f'ibl_choice_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# dataset block config
data_block_path = os.path.join(base_path, 'dataset', 'ibl_block.yaml')
with open(data_block_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data[0]['selection'][0]['dandiset'] = f'ibl_{eid}'
data[0]['selection'][0]['sortsets'][0] = f'{eid}'
with open(os.path.join(base_path, 'dataset', f'ibl_block_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# dataset wheel config
data_wheel_path = os.path.join(base_path, 'dataset', 'ibl_wheel.yaml')
with open(data_wheel_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data[0]['selection'][0]['dandiset'] = f'ibl_{eid}'
data[0]['selection'][0]['sortsets'][0] = f'{eid}'
with open(os.path.join(base_path, 'dataset', f'ibl_wheel_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# dataset whisker config
data_whisker_path = os.path.join(base_path, 'dataset', 'ibl_whisker.yaml')
with open(data_whisker_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data[0]['selection'][0]['dandiset'] = f'ibl_{eid}'
data[0]['selection'][0]['sortsets'][0] = f'{eid}'
with open(os.path.join(base_path, 'dataset', f'ibl_whisker_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# train configs
# train choice config
data_root = os.path.join('/home/ywang74/Dev/poyo_ibl', 'data', 'processed')
train_choice_path = os.path.join(base_path, 'train_ibl_choice.yaml')
with open(train_choice_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data['data_root'] = data_root
data['defaults'][2]['dataset'] = f'ibl_choice_{eid}.yaml'
with open(os.path.join(base_path, f'train_ibl_choice_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# train block config
train_block_path = os.path.join(base_path, 'train_ibl_block.yaml')
with open(train_block_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data['data_root'] = data_root
data['defaults'][2]['dataset'] = f'ibl_block_{eid}.yaml'
with open(os.path.join(base_path, f'train_ibl_block_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# train wheel config
train_wheel_path = os.path.join(base_path, 'train_ibl_wheel.yaml')
with open(train_wheel_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data['data_root'] = data_root
data['defaults'][2]['dataset'] = f'ibl_wheel_{eid}.yaml'
with open(os.path.join(base_path, f'train_ibl_wheel_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)

# train whisker config
train_whisker_path = os.path.join(base_path, 'train_ibl_whisker.yaml')
with open(train_whisker_path, 'r') as file:
    yaml_content = file.read()

data = yaml.safe_load(yaml_content)
data['data_root'] = data_root
data['defaults'][2]['dataset'] = f'ibl_whisker_{eid}.yaml'

with open(os.path.join(base_path, f'train_ibl_whisker_{eid}.yaml'), 'w') as file:
    yaml.dump(data, file)


