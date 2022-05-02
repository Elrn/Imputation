from tensorflow.keras.callbacks import *
import tensorflow as tf
import os
import numpy as np
import re
import logging
from tensorflow.keras import backend

class load_weights(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(load_weights, self).__init__()
        self.filepath = os.fspath(filepath) if isinstance(filepath, os.PathLike) else filepath

    def on_train_batch_begin(self, logs=None):
    # def on_predict_batch_begin(self, logs=None):
    # def on_predict_begin(self, logs=None):
        self.load_weights()

    def load_weights(self):
        filepath_to_load = (self._get_most_recently_modified_file_matching_pattern(self.filepath))
        if (filepath_to_load is not None and self.checkpoint_exists(filepath_to_load)):
            try:
                self.model.load_weights(filepath_to_load)
                print(f'[!] Saved Check point is restored from "{filepath_to_load}".')
            except (IOError, ValueError) as e:
                raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')

    @staticmethod
    def checkpoint_exists(filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return tf.io.gfile.exists(filepath)
        tf_saved_model_exists = tf.io.gfile.exists(filepath)
        tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
            filepath + '.index')
        return tf_saved_model_exists or tf_weights_only_checkpoint_exists

    @staticmethod
    def _get_most_recently_modified_file_matching_pattern(pattern):
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

        latest_tf_checkpoint = tf.train.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if tf.io.gfile.exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (file_path_with_largest_file_name is None or
                            file_path > file_path_with_largest_file_name):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name

class setLR(Callback):
    def __init__(self, lr, verbose=0):
        super(setLR, self).__init__()
        self.lr = lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if not isinstance(self.lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             f'should be float. Got: {self.lr}')
        if isinstance(self.lr, tf.Tensor) and not self.lr.dtype.is_floating:
            raise ValueError(
                f'The dtype of `lr` Tensor should be float. Got: {self.lr.dtype}')
        backend.set_value(self.model.optimizer.lr, backend.get_value(self.lr))
        if self.verbose > 0:
            logging.info(
                f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning '
                f'rate to {self.lr}.')
            logs = logs or {}
            logs['lr'] = backend.get_value(self.model.optimizer.lr)

# plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# early_stopping = EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False
# )
# ckpt = ModelCheckpoint(
#     filepath, monitor='val_loss', verbose=0, save_best_only=False,
#     save_weights_only=False, mode='auto', save_freq='epoch',
# )

