"""Load data, processes it, save it."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)

from one.api import ONE
from data.scripts.ibl_visual_behavior_neuropixels.ibl_data_utils import (
    prepare_data,
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_spike_behavior,
    load_target_behavior,
    load_anytime_behaviors,
    create_intervals, 
)

from kirby.data import (
    Data, 
    IrregularTimeSeries, 
    Interval, 
    DatasetBuilder, 
    ArrayDict,
)
from kirby.taxonomy import (
    Task,
    Sex,
    Species,
    SubjectDescription,
)

SEED = 42
np.random.seed(SEED)

# -------
# SET UP
# -------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str)
ap.add_argument("--base_path", type=str, default="../../")
ap.add_argument("--unit_filter", type=bool, default=False)
ap.add_argument("--unaligned", action="store_true")
args = ap.parse_args()

eid = args.eid
base_path = args.base_path

logging.info(f"Processing session: {eid}")

params = {'interval_len': 1., 'binsize': 0.02, 'single_region': False, 'fr_thresh': 0.5}

if not args.unaligned:
    params.update({'align_time': 'stimOn_times', 'time_window': (-.5, .5)})

STATIC_VARS = ["choice", "reward", "block"]
DYNAMIC_VARS = ["wheel-speed", "whisker-motion-energy", "pupil-diameter"]

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', 
)

# ---------
# Load Data
# ---------
neural_dict, _, meta_data, trials_data, _ = prepare_data(one, eid, params, n_workers=1)
regions, beryl_reg = list_brain_regions(neural_dict, **params)
region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

if args.unaligned:
    intervals = create_intervals(
        start_time=0, 
        end_time=neural_dict["spike_times"].max(), 
        interval_len=params["interval_len"]
    )
    binned_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, 
        neural_dict, 
        intervals=intervals, 
        n_workers=1, 
        **params
    )
else:
    binned_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, 
        neural_dict, 
        trials_df=trials_data['trials_df'], 
        n_workers=1, 
        **params
    )

if args.unit_filter:
    # Filter out inactive neurons
    avg_fr = binned_spikes.sum(1).mean(0) / params['interval_len']
    active_neuron_ids = np.argwhere(avg_fr > 1/params['fr_thresh']).flatten()
else:
    active_neuron_ids = np.arange(binned_spikes.shape[-1])

# -------------------------
# Extract Spiking Activity
# -------------------------
logging.info(f"Extracting spikes ...")

spike_times = neural_dict['spike_times']
spike_clusters = neural_dict['spike_clusters']
unit_mask = np.isin(spike_clusters, active_neuron_ids)
spike_times = spike_times[unit_mask]
spike_clusters = spike_clusters[unit_mask]
unit_ids = np.array(meta_data['uuids'])[active_neuron_ids]

# Convert to spike data object
unit_meta, timestamps, unit_index = [], [], []
for i in range(len(unit_ids)):

    unit_id = unit_ids[i]
    times = spike_times[spike_clusters == i]
    timestamps.append(times)

    if len(times) > 0:
        unit_index.append([i] * len(times))

    unit_meta.append(
        {
            "id": unit_id,
            "unit_number": i,
            "count": len(times),
            "type": 0,
        }
    )
    
unit_meta_df = pd.DataFrame(unit_meta)  
units = ArrayDict.from_dataframe(
    unit_meta_df,
    unsigned_to_long=True,
)
timestamps = np.concatenate(timestamps)
unit_index = np.concatenate(unit_index)
spikes = IrregularTimeSeries(
    timestamps=timestamps,
    unit_index=unit_index,
    domain="auto",
)
spikes.sort()


# ---------------
# Extract Trials
# ---------------
logging.info(f"Extracting trials ...")

if not args.unaligned:
    trial_mask = trials_data['trials_mask']
    binned_behaviors, behavior_masks = bin_behaviors(
        one, eid, DYNAMIC_VARS, trials_df=trials_data['trials_df'], allow_nans=True, **params
    )
else:
    trial_mask = None
    binned_behaviors, behavior_masks = bin_behaviors(
        one, eid, DYNAMIC_VARS, intervals=intervals, allow_nans=True, **params
    )
align_bin_spikes, align_bin_beh, target_mask = align_spike_behavior(
    binned_spikes, binned_behaviors, list(binned_behaviors.keys()), trial_mask, 
)
trial_mask = np.array(target_mask).astype(bool).tolist()

if not args.unaligned:
    trials_data['trials_df'] = trials_data['trials_df'][trial_mask]
    start_time = (
        trials_data['trials_df'][params['align_time']] + params['time_window'][0]
    )
    end_time = (
        trials_data['trials_df'][params['align_time']] + params['time_window'][1]
    )
else:
    intervals = intervals[trial_mask]
    start_time, end_time = (intervals.T[0]), (intervals.T[1])

max_num_trials = sum(trial_mask)
trial_idxs = np.random.choice(np.arange(max_num_trials), max_num_trials, replace=False)
train_idxs = trial_idxs[:int(0.7*max_num_trials)]
val_idxs = trial_idxs[int(0.7*max_num_trials):int(0.8*max_num_trials)]
test_idxs = trial_idxs[int(0.8*max_num_trials):]
trial_split = np.array(['train'] * max_num_trials)
trial_split[val_idxs] = 'val'
trial_split[test_idxs] = 'test'

# Convert to trials data object
trial_table = pd.DataFrame({
    "start": start_time,
    "end": end_time,
    "split_indicator": trial_split,
})
trials = Interval.from_dataframe(trial_table)

train_mask_nwb = trial_table.split_indicator.to_numpy() == "train"
val_mask_nwb = trial_table.split_indicator.to_numpy() == "val"
test_mask_nwb = trial_table.split_indicator.to_numpy() == "test"

trials.train_mask_nwb = (train_mask_nwb)
trials.val_mask_nwb = val_mask_nwb
trials.test_mask_nwb = test_mask_nwb


# -----------------
# Extract Behaviors
# -----------------
logging.info(f"Extracting behaviors ...")

behave_dict = load_anytime_behaviors(one, eid, n_workers=1)

# Extract wheel speed
wh_timestamps = behave_dict["wheel-speed"]["times"]
wh_values = behave_dict["wheel-speed"]["values"].reshape(-1,1)
behavior_type = np.ones_like(wh_timestamps, dtype=np.int64) * 0
eval_mask = np.zeros_like(wh_timestamps, dtype=bool)
for i in range(len(trials)):
    eval_mask[
        (wh_timestamps >= trials.start[i]) & (wh_timestamps < trials.end[i])
    ] = True
wheel = IrregularTimeSeries(
    timestamps=wh_timestamps,
    values=wh_values,
    subtask_index=behavior_type,
    eval_mask=eval_mask,
    domain=Interval(trials.start[0], trials.end[-1]),
)

# Extract whisker
me_timestamps = behave_dict["whisker-motion-energy"]["times"]
me_values = behave_dict["whisker-motion-energy"]["values"].reshape(-1,1)
behavior_type = np.ones_like(me_timestamps, dtype=np.int64) * 0
eval_mask = np.zeros_like(me_timestamps, dtype=bool)
for i in range(len(trials)):
    eval_mask[
        (me_timestamps >= trials.start[i]) & (me_timestamps < trials.end[i])
    ] = True
whisker = IrregularTimeSeries(
    timestamps=me_timestamps,
    values=me_values,
    subtask_index=behavior_type,
    eval_mask=eval_mask,
    domain=Interval(trials.start[0], trials.end[-1]),
)   

if not args.unaligned:
    # Extract choice 
    def map_choice(data):
        choice_map = {"-1": 0, "1": 1}
        return choice_map[str(int(data))]
    choice = trials_data['trials_df'].choice
    stim_start_times = start_time.to_numpy()
    stim_end_times = end_time.to_numpy()
    # create data object for choice and block
    choice = Interval(
        start=stim_start_times,
        end=stim_end_times,
        choice=choice.apply(map_choice).to_numpy(),
        timestamps=stim_start_times / 2.0 + stim_end_times / 2.0,
        timekeys=["start", "end", "timestamps"],
    )

    # Extract block
    def map_block(data):
        block_map = {"0.2": 0, "0.5": 1, "0.8": 2}
        return block_map[str(data)]
    block = trials_data['trials_df'].probabilityLeft
    block = Interval(
        start=stim_start_times,
        end=stim_end_times,
        block=block.apply(map_block).to_numpy(),
        timestamps=stim_start_times / 2.0 + stim_end_times / 2.0,
        timekeys=["start", "end", "timestamps"],
    )

    # Extract reward
    reward = (trials_data["trials_df"]["rewardVolume"] > 1).astype(int).to_numpy()
    reward = Interval(
        start=stim_start_times,
        end=stim_end_times,
        reward=reward,
        timestamps=stim_start_times / 2.0 + stim_end_times / 2.0,
        timekeys=["start", "end", "timestamps"],
    )


# -----------------
# Save Data Object
# -----------------
logging.info(f"Saving data ...")

raw_folder_path = f'{base_path}/raw/'
if args.unaligned:
    processed_folder_path = f'{base_path}/processed/{eid}_unaligned/'
else:
    processed_folder_path = f'{base_path}/processed/{eid}_aligned/'
os.makedirs(raw_folder_path, exist_ok=True)
os.makedirs(processed_folder_path, exist_ok=True)

db = DatasetBuilder(
    raw_folder_path=raw_folder_path,
    processed_folder_path=processed_folder_path,
    # metadata for the dataset
    experiment_name=eid,
    origin_version='',
    derived_version='',
    source='',
    description='',
)

with db.new_session() as session:

    subject = SubjectDescription(
        id=one.get_details(eid)['subject'],
        species=Species.from_string('MUS_MUSCULUS'),
        sex=Sex.from_string('MALE'),
    )
    session.register_subject(subject)

    # extract experiment metadata
    # recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
    session_id = eid

    # register session
    session.register_session(
        id=session_id,
        recording_date=datetime.today().strftime('%Y%m%d'),
        task=Task.FREE_BEHAVIOR,
    )

    # register sortset
    session.register_sortset(
        id=session_id,
        units=units,
    )

    # register session
    session_start, session_end = (trials.start[0], trials.end[-1])
    
    if not args.unaligned:
        data = Data(
            spikes=spikes, units=units, trials=trials,
            wheel=wheel, whisker=whisker,
            choice=choice, block=block, reward=reward,
            domain=Interval(session_start, session_end),
        )
    else:
        data = Data(
            spikes=spikes, units=units, trials=trials,
            wheel=wheel, whisker=whisker,
            domain=Interval(session_start, session_end),
        )

    session.register_data(data)

    # split and register trials into train, validation and test
    train_trials = trials.select_by_mask(trials.train_mask_nwb)
    valid_trials = trials.select_by_mask(trials.val_mask_nwb)
    test_trials = trials.select_by_mask(trials.test_mask_nwb)

    session.register_split("train", train_trials)
    session.register_split("valid", valid_trials)
    session.register_split("test", test_trials)

    # save data to disk
    session.save_to_disk()

# all sessions added, finish by generating a description file for the entire dataset
db.finish()

