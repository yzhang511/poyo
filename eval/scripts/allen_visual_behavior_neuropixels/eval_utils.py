import os
import sys
import pickle
import logging

import uuid
import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn import preprocessing
from scipy.interpolate import interp1d

import torch

def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def bin_target(
    times, 
    values, 
    start, 
    end, 
    binsize=0.02, 
    length=None,
    n_workers=1, 
):  
    num_chunk = len(start)
    if length is None:
        min_length = min(end - start)
        if min_length < 1:
            length = round(min_length, 1)
        else:
            length = int(min_length)
    num_bin = int(np.ceil(length / binsize))

    start_ids = np.searchsorted(times, start, side="right")
    end_ids = np.searchsorted(times, end, side="left")
    _times_list = [times[s_id:e_id] for s_id, e_id in zip(start_ids, end_ids)]
    _vals_list = [values[s_id:e_id] for s_id, e_id in zip(start_ids, end_ids)]

    times_list = [None for _ in range(len(_times_list))]
    vals_list = [None for _ in range(len(_times_list))]
    valid_mask = [None for _ in range(len(_times_list))]
    skip_reason = [None for _ in range(len(_times_list))]

    @globalize
    def interpolate_func(target):
        chunk_idx, target_time, target_val = target
        target_time, target_val = target_time.squeeze(), target_val.squeeze()

        is_valid, x_interp, y_interp = False, None, None
        
        if len(target_val) == 0:
            skip_reason = "target data not present"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.sum(np.isnan(target_val)) > 0:
            skip_reason = "nans in target data"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.isnan(start[chunk_idx]) or np.isnan(end[chunk_idx]):
            skip_reason = "bad interval data"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.abs(start[chunk_idx] - target_time[0]) > binsize:
            skip_reason = "target data starts too late"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.abs(end[chunk_idx] - target_time[-1]) > binsize:
            skip_reason = "target data ends too early"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason

        is_valid, skip_reason = True, None
        x_interp = np.linspace(start[chunk_idx] + binsize, end[chunk_idx], num_bin)

        if len(target_val.shape) > 1 and target_val.shape[1] > 1:
            y_interp_list = []
            for n in range(target_val.shape[1]):
                y_interp_list.append(
                    interp1d(
                        target_time, target_val[:,n], kind="linear", fill_value="extrapolate"
                    )(x_interp)
                )
            y_interp = np.hstack([y[:, None] for y in y_interp_list])
        else:
            y_interp = interp1d(
                target_time, target_val, kind="linear", fill_value="extrapolate"
            )(x_interp)
        return chunk_idx, is_valid, x_interp, y_interp, skip_reason

    with multiprocessing.Pool(processes=n_workers) as p:
        targets = list(zip(np.arange(num_chunk), _times_list, _vals_list))
        with tqdm(total=num_chunk) as pbar:
            for res in p.imap_unordered(interpolate_func, targets):
                pbar.update()
                valid_mask[res[0]] = res[1]
                times_list[res[0]] = res[2]
                vals_list[res[0]] = res[3]
                skip_reason[res[0]] = res[-1]
        pbar.close()
        p.close()

    times_out = np.array(times_list)[valid_mask]
    values_out = np.array(vals_list)[valid_mask]
    times_out = np.array([x.flatten() for x in times_out])
    values_out = np.array([x.flatten() for x in values_out])
    valid_mask = np.array(valid_mask)
    
    return times_out, values_out, valid_mask, skip_reason

