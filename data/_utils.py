"""

"""
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import os
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

def to_arr(path, save_dir):
    with open(path, 'r') as file:
        filename = os.path.splitext(os.path.basename(path))[0]
        save_path = utils.join_dir([save_dir, filename])

        df = parse_fn(file)
        tdf = pd.Index(df.index, name='time').to_frame(index=False)
        np.savez_compressed(save_path, data=df.to_numpy(), time=tdf.to_numpy())

def time_lag_matrix(nan_map, time):
    """
    단순히 초단위로 계산할 것인가?
    - 지수 약 5부터는 값이 없어지게 된다.
    """
    time_lag = np.zeros_like(nan_map, dtype=float)
    for i in range(1, len(time)):
        time_diff = time[i] - time[i - 1]
        to_nan = time_lag[i - 1, :] + time_diff
        cond = np.equal(nan_map[i - 1, :], True)
        time_lag[i, :] = np.where(cond, to_nan, time_diff)

    return time_lag

def parse_fn(file):
    return


########################################################################################################################