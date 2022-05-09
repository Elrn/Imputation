"""

"""
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import os
from os.path import join
import utils
from functools import reduce

idx_cond = lambda series, values:[series==value for value in values]
multi_idx = lambda arr:reduce(lambda x, y: x | y, arr)

def apply(df, idx, functions):
    if hasattr(functions, '__iter__'):
        for f in functions:
            df.loc[idx, 'value'] = df.loc[idx, 'value'].apply(f)
    else:
        df.loc[idx, 'value'] = df.loc[idx, 'value'].apply(functions)
    return df

def _time_lag_matrix(nan_map, time, nan_val=True):
    """
    단순히 초단위로 계산할 것인가?
    - 지수 약 5부터는 값이 없어지게 된다.

    100 단위로 끊어서 5가 되도록?

    :param nan_map : boolean map consist of boolean value with nan is "True"
    :param nan_val : if boolean_map is not consist of boolean,
                     then nan_val should be specified with corresponding value.
    """
    time_lag = np.zeros_like(nan_map, dtype=float)
    for i in range(1, len(time)):
        time_diff = time[i] - time[i - 1]
        to_nan = time_lag[i - 1, :] + time_diff
        cond = np.equal(nan_map[i - 1, :], nan_val) # if nan_map is
        time_lag[i, :] = np.where(cond, to_nan, time_diff)

    return time_lag

def time_lag_matrix(mask_val=np.nan):
    def main(data, time):
        boolean_map = tf.math.is_nan(data)
        time_lag = tf.zeros_like(boolean_map, dtype=data.dtype)
        for i in range(1, time.shape[1]):
            time_diff = time[:, i] - time[:, i - 1]
            to_nan = time_lag[:, i - 1] + time_diff
            cond = tf.equal(boolean_map[:, i - 1, :], mask_val)  # if nan_map is
            time_lag[i, :] = tf.where(cond, to_nan, time_diff)

        return time_lag
    return main

def time_series_crop_n_pad(seq_len, seq_stride=1, start_idx=0, end_idx=None, pad=True):
    seq_idx_fn = np.vectorize(lambda x: slice(x, x + seq_len, 1))

    def main(x):
        if len(x.shape) != 2:
            print(f'Array rank must assigned with 2 (times, features).')
            return
        if x.shape[0] < seq_len:
            if pad != None:
                padding = seq_len - x.shape[0]
                x = np.pad(x, ((0, padding), (0, 0)))
                # print(f'Padded "{x.shape[0]}" to "{x.shape[0]}".')
            else:
                # print(f'sequence is shorter than target sequence lenth.')
                return
        if end_idx == None:
            end_index = x.shape[0]
        num_seqs = end_index - seq_len + 1
        start_indices = np.arange(0, num_seqs, seq_stride, dtype='int32')
        if len(start_indices) == 0:
            return
        slices = seq_idx_fn(start_indices)
        output = np.stack([x[i] for i in slices], 0)
        return output
    return main


########################################################################################################################