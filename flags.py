import os
from os.path import join, dirname
from absl import flags, logging

import utils

# logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS

set_log_verv = lambda debug:logging.set_verbosity(logging.DEBUG) if FLAGS.debug else logging.set_verbosity(logging.INFO)


"""
flags.register_validator('flag',
                         lambda value: value % 2 == 0,
                         message='some message when assert on')
flags.mark_flag_as_required('is_training')
"""


########################################################################################################################
""" model setting """
########################################################################################################################
flags.DEFINE_float('lambda_', 1.0, 'lambda hyperparam for generator loss function', lower_bound=0., upper_bound=1.)

########################################################################################################################
""" Training settings """
########################################################################################################################
flags.DEFINE_boolean('train', True, '모델 학습을 위한 모드')

flags.DEFINE_integer("epochs", 10000, "")
flags.DEFINE_integer('save_frequency', 1, 'save frequency during training', lower_bound=1)

########################################################################################################################
""" Dataset Setting """
########################################################################################################################
flags.DEFINE_integer('input_dims', 41, '', lower_bound=0)
flags.DEFINE_integer('features', 36, '', lower_bound=0)
flags.DEFINE_float('validation_split', 0., 'validation_split', lower_bound=0., upper_bound=0.9)
flags.DEFINE_integer("bsz", 64, "")


########################################################################################################################
""" Optimizer Setting """
########################################################################################################################
flags.DEFINE_float('lr', 1e-5, 'learning rate', upper_bound=1e-3)

########################################################################################################################
""" Directory """
########################################################################################################################
base_dir = os.path.dirname(os.path.realpath(__file__))  # getcwd()
log_dir = utils.join_dir([base_dir, 'log'])
base_ = lambda dir:join(base_dir, dir)
log_ = lambda dir:join(log_dir, dir)

flags.DEFINE_string('home_dir', os.path.expanduser('~'), 'home directory')
flags.DEFINE_string('data_dir', 'C:\\dataset\\PhysioNet\\2012\\set', '')
flags.DEFINE_string('arr_dir', 'C:\\dataset\\PhysioNet\\2012\\arr', '')
flags.DEFINE_string('ckpt_dir', log_('checkpoint'), '체크포인트/모델 저장 경로')
flags.DEFINE_string('plot_dir', log_('plot'), 'plot 저장 경로')
flags.DEFINE_string('result_dir', log_('result'), '')

# flags.DEFINE_string('_log_dir', base_('log/'), '')

########################################################################################################################
