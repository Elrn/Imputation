from absl import app
import tensorflow as tf

import os, re
import utils
import models
import losses
import callbacks
from data import Physionet_2012
from os.path import join
import flags
FLAGS = flags.FLAGS
from tensorflow.python.platform import tf_logging as logging

def main(argv):
    ### ckpt
    ckpt_file_name = 'EP_{epoch}, L_{d_loss:.4f}, vP_{g_loss:.4f}, GP_{GP:.4f}.hdf5'
    ckpt_file_path = join(FLAGS.ckpt_dir, ckpt_file_name)

    ### Get Data
    file_path = join(FLAGS.data_dir, 'PhysioNet2012.npz')
    dataset, _ = Physionet_2012.build(file_path, FLAGS.bsz, 0)

    ### Build model
    # logging.info(f'Build Model ..')
    input_shape = Physionet_2012.get_input_shape()
    model = models.GAN(
        discriminator=models.build(input_shape, models.discriminator(FLAGS.units)),
        generator=models.build(input_shape, models.generator(FLAGS.units))
    )

    ### Compile model
    logging.info(f'Compile Model ..')
    metric_list = [
    ]
    model.compile(
        d_opt=tf.keras.optimizers.Adam(learning_rate=FLAGS.lr),
        g_opt=tf.keras.optimizers.Adam(learning_rate=FLAGS.lr),
        # loss=losses.MSE(),
        # metrics=metric_list,
    )

    ### load weights
    filepath_to_load = callbacks.load_weights._get_most_recently_modified_file_matching_pattern(ckpt_file_path)
    if (filepath_to_load is not None and callbacks.load_weights.checkpoint_exists(filepath_to_load)):
        initial_epoch = int(re.findall(r"EP_(\d+),", filepath_to_load)[0])
        try:
            model.load_weights(filepath_to_load)
            print(f'[Model|ckpt] Saved Check point is restored from "{filepath_to_load}".')
        except (IOError, ValueError) as e:
            raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')
    else:
        print(f'[Model|ckpt] Model is trained from scratch.')
        initial_epoch = 0

    ### Train model
    logging.info(f'Model fit ..')
    history = model.fit(
        x=dataset,
        epochs=FLAGS.epochs,
        # validation_data=val_dataset,
        initial_epoch=initial_epoch,
        callbacks=[
            callbacks.ModelCheckpoint(ckpt_file_path, monitor='loss', save_best_only=True, save_weights_only=False, save_freq='epoch'),
            # EarlyStopping(monitor='loss', min_delta=0, patience=5),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, min_delta=0.0001, cooldown=0, min_lr=0),
            # callbacks.setLR(0.0001),
        ]
    )
    return

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass