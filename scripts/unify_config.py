import yaml
import os
import argparse

parser = argparse.ArgumentParser(description='Create a config file')
parser.add_argument('--base_path', type=str, help='Base path for the dataset')
parser.add_argument('--eid_list_path', type=str, help='Path to the eid list')

args = parser.parse_args()
base_path = args.base_path
base_path = os.path.join(base_path, 'configs')
eid_list_path = args.eid_list_path
print(f'Base path: {base_path}')
print(eid_list_path)
with open(eid_list_path, 'r') as file:
    eid_list = file.read().splitlines()
all_eid_content = []
for eid in eid_list:
    data_eid_path = os.path.join(base_path, 'dataset', f'ibl_multitask_{eid}.yaml')
    print(data_eid_path)
    with open(data_eid_path, 'r') as file:
        yaml_content = file.read()
    data = yaml.safe_load(yaml_content)
    all_eid_content.extend(data)
print('Creating unified config')
# save the all_eid_content to a yaml file
with open(os.path.join(base_path, 'dataset', 'ibl_sessions.yaml'), 'w') as file:
    yaml.dump(all_eid_content, file)