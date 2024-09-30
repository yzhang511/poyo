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
                choice_res[eid] = data['Choice']
            elif 'block' in f:
                block_res[eid] = data['Block']
            elif 'wheel' in f:
                wheel_res[eid] = data['Wheel']
            elif 'whisker' in f:
                whisker_res[eid] = data['Whisker']
            else:
                raise ValueError('Unknown result type')
print('Choice / Block / Wheel / Whisker')
print("Accuracy Result")
choice_list, block_list, wheel_list, whisker_list = [], [], [], []
for eid in test_eids:
    try:
        print(f'{eid}: {round(choice_res[eid]["accuracy"],5)} / {round(block_res[eid]["accuracy"],5)} / {round(wheel_res[eid]["r2_trial"],5)} / {round(whisker_res[eid]["r2_trial"],5)}')
        choice_list.append(choice_res[eid]["accuracy"])
        block_list.append(block_res[eid]["accuracy"])
        wheel_list.append(wheel_res[eid]["r2_trial"])
        whisker_list.append(whisker_res[eid]["r2_trial"])
    except:
        continue
print(f"Mean: {round(np.mean(choice_list),5)} / {round(np.mean(block_list),5)} / {round(np.mean(wheel_list),5)} / {round(np.mean(whisker_list),5)}")

print("Balanced Accuracy Result")
choice_list, block_list, wheel_list, whisker_list = [], [], [], []
for eid in test_eids:
    try:
        print(f'{eid}: {round(choice_res[eid]["balanced_accuracy"],5)} / {round(block_res[eid]["balanced_accuracy"],5)} / {round(wheel_res[eid]["r2_trial"],5)} / {round(whisker_res[eid]["r2_trial"],5)}')
        choice_list.append(choice_res[eid]["balanced_accuracy"])
        block_list.append(block_res[eid]["balanced_accuracy"])
        wheel_list.append(wheel_res[eid]["r2_trial"])
        whisker_list.append(whisker_res[eid]["r2_trial"])
    except:
        continue
print(f"Mean: {round(np.mean(choice_list),5)} / {round(np.mean(block_list),5)} / {round(np.mean(wheel_list),5)} / {round(np.mean(whisker_list),5)}")