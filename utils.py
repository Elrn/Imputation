import os
import logging
import tensorflow as tf


def tf_init():
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True
    logging.getLogger('h5py._conv').disabled = True

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def join(dirs:list):
    if len(dirs) == 0:
        return dirs
    base = dirs[0]
    for dir in dirs[1:]:
        base = os.path.join(base, dir)
    return base
#
def mkdir(path):
    try: # if hasattr(path, '__len__') and type(path) != str:
        os.makedirs(path)
    except OSError as error:
        print(error)