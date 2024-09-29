import os
import numpy as np
import glob
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--result_dir', type=str, default='results')

args = argparser.parse_args()

with open('data/test_eids.txt') as file:
    test_eids = [line.rstrip() for line in file]

result_dir = args.result_dir
result_files = glob.glob(os.path.join(result_dir, '*/*.npy'))
result_files = [f for f in result_files if any(eid in f for eid in test_eids)]
choice_res, block_res, wheel_res, whisker_res = {}, {}, {}, {}
for eid in test_eids:
    for f in result_files:
        if eid in f:
            data = np.load(f, allow_pickle=True).item()
            if 'choice' in f:   
                choice_res[eid] = data['Block']
            elif 'block' in f:
                block_res[eid] = data['Block']
            elif 'wheel' in f:
                wheel_res[eid] = data['Block']
            elif 'whisker' in f:
                whisker_res[eid] = data['Block']
            else:
                raise ValueError('Unknown result type')
            
for eid in test_eids:
    print(f'Choice: {eid}, {choice_res[eid]}')
    print(f'Block: {eid}, {block_res[eid]}')
    print(f'Wheel: {eid}, {wheel_res[eid]}')
    print(f'Whisker: {eid}, {whisker_res[eid]}')
    print('')