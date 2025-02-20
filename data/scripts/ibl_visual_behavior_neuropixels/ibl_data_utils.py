import os 
import sys
import uuid
from tqdm import *
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from functools import partial
from scipy.interpolate import interp1d
from iblutil.numerical import ismember, bincount2D
import brainbox.behavior.dlc as dlc
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.regions import BrainRegions
from brainbox.population.decode import get_spike_counts_in_bins

DYNAMIC_VARS = [
    "wheel-speed", "whisker-motion-energy", "body-motion-energy", 
]

def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result


def load_spiking_data(one, pid, compute_metrics=False, qc=None, **kwargs):
    eid = kwargs.pop("eid", "")
    pname = kwargs.pop("pname", "")
    sampling_freq = 30_000
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    
    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics
    )
    if clusters_labeled is None:
        return None, None, None
    else:
        clusters_labeled = clusters_labeled.to_df()
    
    if qc is None:
        return spikes, clusters_labeled, sampling_freq
    else:
        iok = clusters_labeled["label"] >= qc
        selected_clusters = clusters_labeled[iok]
        spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index)
        selected_clusters.reset_index(drop=True, inplace=True)
        selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
        selected_spikes["clusters"] = selected_clusters.index[ib].astype(np.int32)
        return selected_spikes, selected_clusters, sampling_freq


def merge_probes(spikes_list, clusters_list):
    assert (len(clusters_list) == len(spikes_list)), \
        "clusters_list and spikes_list must have the same length"
    assert all([isinstance(s, dict) for s in spikes_list]), \
        "spikes_list must contain only dictionaries"
    assert all([isinstance(c, pd.DataFrame) for c in clusters_list]), \
        "clusters_list must contain only pd.DataFrames"

    merged_spikes, merged_clusters = [], []
    cluster_max = 0

    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes["clusters"] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
        
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {
        k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()
    }
    sort_idx = np.argsort(merged_spikes["times"], kind="stable")
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
    return merged_spikes, merged_clusters


def load_trials_and_mask(
    one, 
    eid, 
    min_rt=0.08, 
    max_rt=2., 
    nan_exclude="default", 
    min_trial_len=None,
    max_trial_len=None, 
    exclude_unbiased=False, 
    exclude_nochoice=True, 
    sess_loader=None,
): 
    if nan_exclude == "default":
        nan_exclude = [
            "stimOn_times",
            "choice",
            "feedback_times",
            "probabilityLeft",
            "firstMovement_times",
            "feedbackType"
        ]

    if sess_loader is None:
        sess_loader = SessionLoader(one, eid)

    if sess_loader.trials.empty:
        sess_loader.load_trials()

    if min_rt is not None:
        query = f"(firstMovement_times - stimOn_times < {min_rt})"
    else:
        query = ""
    if max_rt is not None:
        query += f" | (firstMovement_times - stimOn_times > {max_rt})"
    if min_trial_len is not None:
        query += f" | (feedback_times - goCue_times < {min_trial_len})"
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    if exclude_unbiased:
        query += " | (probabilityLeft == 0.5)"
    if exclude_nochoice:
        query += " | (choice == 0)"
    if min_rt is None:
        query = query[3:]

    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask


def list_brain_regions(neural_dict, **kwargs):
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict["cluster_regions"], mapping="Beryl")
    regions = (
        [[k] for k in np.unique(beryl_reg)] if kwargs["single_region"] else [np.unique(beryl_reg)]
    )
    print(f"Use spikes from brain regions: ", regions[0])
    return regions, beryl_reg


def select_brain_regions(regressors, beryl_reg, region, **kwargs):
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def get_spike_data_per_interval(
    times, 
    clusters, 
    interval_begs, 
    interval_ends, 
    interval_len, 
    binsize, 
    n_workers=os.cpu_count()
):
    n_intervals = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    @globalize
    def compute_spike_count(interval):
        interval_idx, t_beg, t_end = interval
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        return binned_spikes_tmp[:, :n_bins], idxs_tmp, interval_idx

    binned_spikes = np.zeros((n_intervals, n_clusters_in_region, n_bins))
    with multiprocessing.Pool(processes=n_workers) as p:
        intervals = list(zip(np.arange(n_intervals), interval_begs, interval_ends))
        with tqdm(total=len(intervals)) as pbar:
            for res in p.imap_unordered(compute_spike_count, intervals):
                pbar.update()
                binned_spikes[res[-1], res[1], :] += res[0]
        pbar.close()
        p.close()
    return binned_spikes


def bin_spiking_data(
    reg_clu_ids, 
    neural_df, 
    intervals=None, 
    trials_df=None, 
    n_workers=os.cpu_count(), 
    **kwargs
):
    if trials_df is not None:
        intervals = np.vstack([
            trials_df[kwargs["align_time"]] + kwargs["time_window"][0],
            trials_df[kwargs["align_time"]] + kwargs["time_window"][1]
        ]).T
        chunk_len = kwargs["time_window"][1] - kwargs["time_window"][0]
        interval_len = (
            kwargs["time_window"][1] - kwargs["time_window"][0]
        )
    else:
        assert intervals is not None, \
            "Require intervals to segment the recording into chunks including trials and non-trials."
        chunk_len = intervals[0,1] - intervals[0,0]
        interval_len = (
            intervals[0,1] - intervals[0,0]
        )
    # subselect spikes for this region
    spikemask = np.isin(neural_df["spike_clusters"], reg_clu_ids)
    regspikes = neural_df["spike_times"][spikemask]
    regclu = neural_df["spike_clusters"][spikemask]
    clusters_used_in_bins = np.unique(regclu)
    binsize = kwargs.get("binsize", chunk_len)
    
    if chunk_len / binsize == 1.0:
        # one vector of neural activity per interval
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        binned_array = get_spike_data_per_interval(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs["binsize"],
            n_workers=n_workers
        )
        binned_list = [x.T for x in binned_array]   
    return np.array(binned_list), clusters_used_in_bins

def load_target_behavior(one, eid, target):
    """
    Parameters
    ----------
    target : str
        'wheel-position' | 'wheel-velocity' | 'wheel-speed' | 
        'left-whisker-motion-energy' | 'right-whisker-motion-energy' | 
        'left-pupil-diameter' | 'right-pupil-diameter' |
        'left-camera-left-paw-speed' | 'left-camera-right-paw-speed' | 
        'right-camera-left-paw-speed' | 'right-camera-right-paw-speed' |
        'left-nose-speed' | 'right-nose-speed'
    one : 
    eid : str

    Returns
    -------
    dict
        'times': timestamps for behavior signal
        'values': associated values
        'skip': bool, True if there was an error loading data
    """

    # To load wheel and motion energy, we just use the SessionLoader, e.g.
    sess_loader = SessionLoader(one, eid)
    
    # wheel is a dataframe that contains wheel times and position interpolated to a uniform sampling rate, velocity and
    # acceleration computed using Gaussian smoothing
    try:
        if target == 'wheel-position':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['position'].to_numpy()
            }
        elif target == 'wheel-velocity':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['velocity'].to_numpy()
            }
        elif target == 'wheel-speed':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': np.abs(sess_loader.wheel['velocity'].to_numpy())
            }
        # motion_energy is a dictionary of dataframes, each containing the times and the motion energy for each view
        # for the side views, they contain columns ['times', 'whiskerMotionEnergy'] for the body view it contains
        # ['times', 'bodyMotionEnergy']
        elif target == 'left-whisker-motion-energy':
            sess_loader.load_motion_energy(views=['left'])
            beh_dict = {
                'times': sess_loader.motion_energy['leftCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy'].to_numpy()
            }
        elif target == 'right-whisker-motion-energy':
            sess_loader.load_motion_energy(views=['right'])
            beh_dict = {
                'times': sess_loader.motion_energy['rightCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['rightCamera']['whiskerMotionEnergy'].to_numpy()
            }
        elif target == 'left-pupil-diameter':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc_left.features.pupilDiameter_smooth
            }
        elif target == 'right-pupil-diameter':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc_right.features.pupilDiameter_smooth
            }
        elif target == 'left-camera-left-paw-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_l")
            }
        elif target == 'left-camera-right-paw-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_r")
            }
        elif target == 'right-camera-left-paw-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_l")
            }
        elif target == 'right-camera-right-paw-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_r")
            }
        elif target == 'left-nose-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="nose_tip")
            }
        elif target == 'right-nose-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="nose_tip")
            }
        else:
            raise NotImplementedError
    except BaseException as e:
        print('Error loading %s data' % target)
        print(e)
        beh_dict = {'times': None, 'values': None, 'skip': True}
 
    return beh_dict


def get_behavior_per_interval(
    target_times, 
    target_vals, 
    intervals=None, 
    trials_df=None, 
    allow_nans=False, 
    n_workers=os.cpu_count(), 
    **kwargs
):
    """
    Format a single session-wide array of target data into a list of interval-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `align_interval`; for example if `align_interval=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `align_interval=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_vals : array-like
        data samples
    intervals : 
        array of time intervals for each recording chunk including trials and non-trials
    trials_df : pd.DataFrame
        requires a column that matches `align_event`
    align_event : str
        event to align interval to
        firstMovement_times | stimOn_times | feedback_times
    align_interval : tuple
        (align_begin, align_end); time in seconds relative to align_event
    binsize : float
        size of individual bins in interval
    allow_nans : bool, optional
        False to skip intervals with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each interval
        - (list): data for each interval
        - (array-like): mask of good intervals (True) and bad intervals (False)

    """

    binsize = kwargs['binsize']
    interval_len = kwargs['interval_len']

    if trials_df is not None:
        align_event = kwargs['align_time']
        align_times = trials_df[align_event].values
        align_interval = kwargs['time_window']
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        interval_begs, interval_ends = intervals.T

    n_intervals = len(interval_begs)

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print('interval times all nan')
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        target_times_list = []
        target_vals_list = []
        return target_times_list, target_vals_list, good_interval

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = [None for _ in range(len(target_times_og_list))]
    target_vals_list = [None for _ in range(len(target_times_og_list))]
    good_interval = [None for _ in range(len(target_times_og_list))]
    skip_reasons = [None for _ in range(len(target_times_og_list))]

    @globalize
    def interpolate_behavior(target):
        # We use interval_idx to track the interval order while working with p.imap_unordered()
        interval_idx, target_time, target_vals = target

        is_good_interval, x_interp, y_interp = False, None, None
        
        if len(target_vals) == 0:
            skip_reason = 'target data not present'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.sum(np.isnan(target_vals)) > 0 and not allow_nans:
            skip_reason = 'nans in target data'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.isnan(interval_begs[interval_idx]) or np.isnan(interval_ends[interval_idx]):
            skip_reason = 'bad interval data'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.abs(interval_begs[interval_idx] - target_time[0]) > binsize:
            skip_reason = 'target data starts too late'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.abs(interval_ends[interval_idx] - target_time[-1]) > binsize:
            skip_reason = 'target data ends too early'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

        is_good_interval, skip_reason = True, None
        x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)
        if len(target_vals.shape) > 1 and target_vals.shape[1] > 1:
            n_dims = target_vals.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(interp1d(
                    target_time, target_vals[:, n], kind='linear',
                    fill_value='extrapolate')(x_interp))
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(
                target_time, target_vals, kind='linear', fill_value='extrapolate')(x_interp)
        return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

    with multiprocessing.Pool(processes=n_workers) as p:
        targets = list(zip(np.arange(n_intervals), target_times_og_list, target_vals_og_list))
        with tqdm(total=n_intervals) as pbar:
            for res in p.imap_unordered(interpolate_behavior, targets):
                pbar.update()
                good_interval[res[0]] = res[1]
                target_times_list[res[0]] = res[2]
                target_vals_list[res[0]] = res[3]
                skip_reasons[res[0]] = res[-1]
        pbar.close()
        p.close()

    return target_times_list, target_vals_list, np.array(good_interval), skip_reasons    


def load_anytime_behaviors(one, eid, n_workers=os.cpu_count()):

    behaviors = [
        "wheel-speed",
        "whisker-motion-energy", 
        "body-motion-energy",
        "pupil-diameter", 
        # "wheel-velocity", 
    ]
    @globalize
    def load_beh(beh):
        if beh == "whisker-motion-energy":
            target_dict = load_target_behavior(one, eid, "left-whisker-motion-energy")
            if "skip" in target_dict.keys():
                target_dict = load_target_behavior(one, eid, "right-whisker-motion-energy")
        elif beh == "pupil-diameter":
            target_dict = load_target_behavior(one, eid, "left-pupil-diameter")
            if "skip" in target_dict.keys():
                target_dict = load_target_behavior(one, eid, "right-pupil-diameter")
        else:
            target_dict = load_target_behavior(one, eid, beh)
        return beh, target_dict
    
    behave_dict = {}
    with multiprocessing.Pool(processes=n_workers) as p:
        with tqdm(total=len(behaviors)) as pbar:
            for res in p.imap_unordered(load_beh, behaviors):
                pbar.update()
                behave_dict.update({res[0]: res[1]})
        pbar.close()
        p.close()
    return behave_dict


def bin_behaviors(
    one, 
    eid, 
    behaviors,
    intervals=None, 
    trials_only=False, 
    trials_df=None, 
    mask=None, 
    allow_nans=True, 
    n_workers=os.cpu_count(),
    **kwargs
):

    behave_dict, mask_dict = {}, {}
    
    if mask is not None:
        trials_df = trials_df[mask]

    if trials_df is not None:        
        choice = trials_df['choice'].to_numpy()
        reward = (trials_df['rewardVolume'] > 1).astype(int).to_numpy()
        contrast = np.c_[trials_df['contrastLeft'], trials_df['contrastRight']]
        contrast = np.nan_to_num(contrast, 0)
        stimside = np.argmax(contrast, 1)
        contrast = contrast.sum(1)

        behave_dict.update(
            {'choice': choice, 'stimside': stimside, 'reward': reward, 'contrast': contrast}
        )
        behave_mask = np.ones(len(trials_df)) 
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        behave_mask = np.ones(len(intervals)) 
        
    for beh in behaviors:
        if beh == 'whisker-motion-energy':
            target_dict = load_target_behavior(one, eid, 'left-whisker-motion-energy')
            if 'skip' in target_dict.keys():
                target_dict = load_target_behavior(one, eid, 'right-whisker-motion-energy')  
        elif beh == 'pupil-diameter':
            target_dict = load_target_behavior(one, eid, 'left-pupil-diameter')
            if 'skip' in target_dict.keys():
                target_dict = load_target_behavior(one, eid, 'right-pupil-diameter')  
        else:
            target_dict = load_target_behavior(one, eid, beh)
        target_times, target_vals = target_dict['times'], target_dict['values']
        target_times_list, target_vals_list, target_mask, skip_reasons = get_behavior_per_interval(
            target_times, target_vals, intervals=intervals, 
            trials_df=trials_df, allow_nans=allow_nans, n_workers=n_workers, **kwargs
        )
        behave_dict.update({beh: np.array(target_vals_list, dtype=object)})
        mask_dict.update({beh: target_mask})
        behave_mask = np.logical_and(behave_mask, target_mask)

    if not allow_nans:
        for k, v in behave_dict.items():
            behave_dict[k] = behave_dict[beh][behave_mask]
    
    return behave_dict, mask_dict


def prepare_data(one, eid, params, n_workers=os.cpu_count()):
    pids, probe_names = one.eid2pid(eid) 
    details = one.get_details(eid)
    print(f"Merge {len(probe_names)} probes for session EID: {eid}")

    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters, sampling_freq = load_spiking_data(
            one, pid, eid=eid, pname=probe_name
        )
        if tmp_spikes is None:
            return None, None, None, None, None
        tmp_clusters["pid"] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    trials_df, trials_mask = load_trials_and_mask(
        one=one, eid=eid, max_trial_len=10.0, 
    )
        
    behave_dict = load_anytime_behaviors(one, eid, n_workers=n_workers)
    
    neural_dict = {
        "spike_times": spikes["times"],
        "spike_clusters": spikes["clusters"],
        "cluster_regions": clusters["acronym"].to_numpy(),
    }
        
    meta_data = {
        "eid": eid,
        "subject": details["subject"],
        "lab": details["lab"],
        "sampling_freq": sampling_freq,
        "cluster_channels": list(clusters["channels"]),
        "cluster_regions": list(clusters["acronym"]),
        "good_clusters": list((clusters["label"] >= 1).astype(int)),
        "cluster_depths": list(clusters["depths"]),
        "uuids":  list(clusters["uuids"]),
        # "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
    }

    trials_data = {
        "trials_df": trials_df,
        "trials_mask": trials_mask
    }
    return neural_dict, behave_dict, meta_data, trials_data, trials_mask


def create_intervals(start_time, end_time, interval_len):
    interval_begs = np.arange(
        start_time, end_time-interval_len, interval_len
    )
    interval_ends = np.arange(
        start_time+interval_len, end_time, interval_len
    )
    return np.c_[interval_begs, interval_ends]


def align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask=None):
    """Function to verify trial alignment between neural and behavior data.
    """
    target_mask = [1] * len(binned_spikes)
    for beh_name in beh_names:
        beh_mask = [1 if trial is not None else 0 for trial in binned_behaviors[beh_name]]
    target_mask = target_mask and beh_mask

    if trials_mask is not None:
        if not isinstance(trials_mask, np.ndarray):
            trials_mask = trials_mask.to_numpy()
        target_mask = target_mask and list(trials_mask.astype(int))

    del_idxs = np.argwhere(np.array(target_mask) == 0)

    aligned_binned_spikes = np.delete(binned_spikes, del_idxs, axis=0)

    aligned_binned_behaviors = {}
    for beh_name in beh_names:
        aligned_binned_behaviors.update({beh_name: np.delete(binned_behaviors[beh_name], del_idxs, axis=0)})
        aligned_binned_behaviors[beh_name] = np.array(
                [y for y in aligned_binned_behaviors[beh_name]], dtype=float
            ).reshape((aligned_binned_spikes.shape[0], -1)
        )
        assert len(aligned_binned_spikes) == len(aligned_binned_behaviors[beh_name]), f'mismatch between spike shape {len(aligned_binned_spikes)} and {beh_name} shape {len(aligned_binned_behaviors[beh_name])}'
    
    return aligned_binned_spikes, aligned_binned_behaviors, target_mask

