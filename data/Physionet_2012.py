"""
Classification of Heart Sound Recordings:
    The PhysioNet/Computing in Cardiology Challenge 2016

https://physionet.org/content/challenge-2016/1.0.0/

------------------------------------------------------------------------------------------------------------------------
[ General Descriptors ]

RecordID    (a unique integer for each ICU stay)
Age         (years)
Gender      (0: female, or 1: male)
Height      (cm)
ICUType     (1: Coronary Care Unit
             2: Cardiac Surgery Recovery Unit
             3: Medical ICU
             4: Surgical ICU)
Weight      (kg) *.

------------------------------------------------------------------------------------------------------------------------
[ Time series fearues ]

Albumin         (g/dL)
ALP             [Alkaline phosphatase (IU/L)]
ALT             [Alanine transaminase (IU/L)]
AST             [Aspartate transaminase (IU/L)]
Bilirubin       (mg/dL)
BUN             [Blood urea nitrogen (mg/dL)]
Cholesterol     (mg/dL)
Creatinine      [Serum creatinine (mg/dL)]
DiasABP         [Invasive diastolic arterial blood pressure (mmHg)]
FiO2            [Fractional inspired O2 (0-1)]
GCS             [Glasgow Coma Score (3-15)]
Glucose         [Serum glucose (mg/dL)]
HCO3            [Serum bicarbonate (mmol/L)]
HCT             [Hematocrit (%)]
HR              [Heart rate (bpm)]
K               [Serum potassium (mEq/L)]
Lactate         (mmol/L)
Mg              [Serum magnesium (mmol/L)]
MAP             [Invasive mean arterial blood pressure (mmHg)]
MechVent        [Mechanical ventilation respiration (0:false, or 1:true)]
Na              [Serum sodium (mEq/L)]
NIDiasABP       [Non-invasive diastolic arterial blood pressure (mmHg)]
NIMAP           [Non-invasive mean arterial blood pressure (mmHg)]
NISysABP        [Non-invasive systolic arterial blood pressure (mmHg)]
PaCO2           [partial pressure of arterial CO2 (mmHg)]
PaO2            [Partial pressure of arterial O2 (mmHg)]
pH              [Arterial pH (0-14)]
Platelets       (cells/nL)
RespRate        [Respiration rate (bpm)]
SaO2            [O2 saturation in hemoglobin (%)]
SysABP          [Invasive systolic arterial blood pressure (mmHg)]
Temp            [Temperature (°C)]
TropI           [Troponin-I (μg/L)]
TropT           [Troponin-T (μg/L)]
Urine           [Urine output (mL)]
WBC             [White blood cell count (cells/nL)]
Weight          (kg)*

------------------------------------------------------------------------------------------------------------------------
[ Outcome-related Descriptors ]

RecordID            (defined as above)
SAPS-I score        (Le Gall et al., 1984)
SOFA score          (Ferreira et al., 2001)
Length of stay      (days)
Survival            (days)
In-hospital death   (0: survivor, or 1: died in-hospital)

------------------------------------------------------------------------------------------------------------------------
"""
import os.path

import numpy as np
import tensorflow as tf
import flags
import utils
import data._utils as dutils
from absl import app
from glob import glob
import pandas as pd

FLAGS = flags.FLAGS

num_columns = 36 # except header cols
########################################################################################################################
get_input_shape = lambda :[FLAGS.input_dims, num_columns]

def parse_fn(x):
    x = tf.transpose(x, [1,0])
    return x

def validation_split_fn(dataset, validation_split):
    len_dataset = tf.data.experimental.cardinality(dataset).numpy()
    valid_count = int(len_dataset * validation_split)
    print(f'[Dataset|split] Total: "{len_dataset}", Train: "{len_dataset-valid_count}", Valid: "{valid_count}"')
    return dataset.skip(valid_count), dataset.take(valid_count)

def build(file_pattern, bsz, validation_split=0.1):
    assert 0 <= validation_split <= 0.5
    paths = glob(file_pattern)
    arrs = []
    for path in paths:
        arrs.extend(np.load(path))

    dataset = load(arrs, bsz)
    if validation_split is not None and validation_split is not 0:
        return validation_split_fn(dataset, validation_split)
    else:
        return dataset, None

def load(arrs, bsz, drop=True):
    return tf.data.Dataset.from_tensor_slices(
        arrs,
    ).prefetch(
        tf.data.experimental.AUTOTUNE
    ).map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).interleave(
        lambda x : tf.keras.utils.timeseries_dataset_from_array(
            x, targets=None, sequence_length=FLAGS.input_dims,
        # ).map(
        #     parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ),
        cycle_length = tf.data.experimental.AUTOTUNE,
        num_parallel_calls = tf.data.experimental.AUTOTUNE
    # ).repeat(
    #     count=3
    # ).shuffle(
    #     4,
    #     reshuffle_each_iteration=True
    ).cache(
    ).batch(
        batch_size=bsz,
        drop_remainder=drop,
    )
########################################################################################################################
base_features = {
    'SaO2', 'K', 'NISysABP', 'Lactate', 'PaCO2', 'SysABP',
    'Albumin', 'FiO2', 'TroponinI', 'ALP', 'TroponinT',
    'Creatinine', 'NIMAP', 'DiasABP', 'NIDiasABP', 'ALT',
    'HR', 'Glucose', 'Mg', 'PaO2', 'MAP',
    'AST', 'Temp', 'GCS', 'Platelets', 'Bilirubin', 'Cholesterol',
    'Na', 'Urine', 'BUN', 'WBC', 'HCT', 'HCO3', 'MechVent', 'RespRate', 'pH'
}

def refinement(df):
    series = df['parameter']
    # --------------------------------------------------------------------------------------------------
    idx = series == 'DiasABP'
    functions = [
        lambda x: np.nan if x == -1 else x,
        lambda x: np.nan if x < 1. else x,
        lambda x: np.nan if x > 200 else x,
    ]
    df = dutils.apply(df, idx, functions)
    # --------------------------------------------------------------------------------------------------
    fs = ['SysABP', 'MAP', 'NIDiasABP', 'NISysABP', 'NIMAP', 'HR', 'PaCO2', 'PaO2', 'RespRate', 'WBC']
    idx = dutils.multi_idx(dutils.idx_cond(series, fs))
    df = dutils.apply(df, idx, lambda x:np.nan if x < 1 else x)
    # --------------------------------------------------------------------------------------------------
    idx = series == 'HR'
    df = dutils.apply(df, idx, lambda x: np.nan if (x > 299) else x)
    # --------------------------------------------------------------------------------------------------
    idx = series == 'PaCO2'
    df = dutils.apply(df, idx, lambda x: x * 10 if x < 10 else x)
    # --------------------------------------------------------------------------------------------------
    idx = series == 'PaO2'
    df = dutils.apply(df, idx, lambda x: x * 10 if x < 20 else x)
    # --------------------------------------------------------------------------------------------------
    idx = series == 'pH'
    functions = [
        lambda x: x * 10 if (x < 0.8 or x > 0.65) else x,
        lambda x: x * 0.1 if (x < 80 or x > 65) else x,
        lambda x: x * 0.01 if (x < 800 or x > 650) else x,
        lambda x: x * 0.01 if (x < 6.5 or x > 8.0) else np.nan
    ]
    df = dutils.apply(df, idx, functions)
    # --------------------------------------------------------------------------------------------------
    idx = series == 'PaO2'
    functions = [
        lambda x: x * 9 / 5 + 32 if (x < 10 or x > 1) else x,
        lambda x: (x - 32) * 5 / 9 if (x < 113 or x > 95) else x,
        lambda x: np.nan if x < 25 else x,
        lambda x: np.nan if x < 45 else x
    ]
    df = dutils.apply(df, idx, functions)
    # --------------------------------------------------------------------------------------------------
    idx = series == 'Weight'
    df = dutils.apply(df, idx, [lambda x: np.nan if x < 35 else x])
    df = dutils.apply(df, idx, [lambda x: np.nan if x > 299 else x])
    # --------------------------------------------------------------------------------------------------
    return df

def main(argv):
    pattern = utils.join_dir([FLAGS.data_dir, '*.txt'])
    paths = glob(pattern)
    save_dir = 'C:\\dataset\\PhysioNet\\2012\\arr'

    for file in paths:
        print(file)
        txt_all = []
        with open(file, 'r') as fp:
            txt = fp.readlines()
        txt = [t.rstrip('\n').split(',') for t in txt]
        txt_all.extend(txt[1:])  # except column name
        df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value'])
        ### time 변환
        df['time'] = df['time'].map(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1]))
        # type 변환
        df['value'] = pd.to_numeric(df['value'], errors='raise')
        #
        df.reset_index(drop=True, inplace=True)  # header를 제외한 index를 처음부터 다시 설정
        ### handling abnormal data
        df = refinement(df)
        ### alignment along time
        df = df.groupby(['time', 'parameter']).last()
        df.reset_index(inplace=True)
        df = df.pivot(index='time', columns='parameter', values='value')
        ### 없는 features 를 np.nan으로 추가
        cur_features = set(df.columns)
        add_features = base_features - cur_features
        df[list(add_features)] = np.nan
        df = df.reindex(sorted(df.columns), axis=1)

        ### Header
        # header = df.loc[df['time'] == 0, :].copy()
        header_features = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
        ### Header 제거
        df = df.drop(columns=header_features)
        # header = header.loc[df['parameter'].isin(header_features)]
        # df = df.loc[~df.index.isin(header.index), :]

        ### Normalization data
        df = (df - df.min()) / (df.max() - df.min())
        ### time matrix
        nan_map = pd.isnull(df).to_numpy()
        tdf = pd.Index(df.index, name='time')#.to_frame(index=False)
        time = dutils.time_lag_matrix(nan_map, tdf)

        filename = os.path.splitext(os.path.basename(file))[0]
        # filename = os.path.dirname(file)
        # save_path = utils.join_dir([save_dir, filename])
        # np.savez_compressed(save_path, data=df.to_numpy(), time=time)
        np.save(utils.join_dir([save_dir, filename + '_data']), df.to_numpy())
        np.save(utils.join_dir([save_dir, filename + '_time']), time)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass