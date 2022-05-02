from absl import app
import os
import utils
from data import Physionet_2012
import flags
import train

FLAGS = flags.FLAGS

def main(argv):
    if argv[0] == __file__:
        utils.tf_init()
    # init
    paths = [FLAGS.ckpt_dir, FLAGS.plot_dir]
    [utils.mkdir(path) for path in paths]

    if FLAGS.train:
        train.main('')
    else:
        return
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass