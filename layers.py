import tensorflow as tf
from keras.layers import recurrent, recurrent_v2
from keras.engine import base_layer
from keras import backend
from tensorflow.python.platform import tf_logging as logging
from keras import constraints
from keras import initializers
from keras import regularizers
from keras import activations

########################################################################################################################
# Dropout 추가 고려
# class GRUICell(recurrent.GRUCell):
#     """
#     https://yjjo.tistory.com/18
#
#     """
#     def __init__(self,
#                  units=None,
#                  activation='tanh',
#                  recurrent_activation='hard_sigmoid',
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros',
#                  dropout=0.,
#                  recurrent_dropout=0.,
#                  reset_after=True,
#                  **kwargs
#                  ):
#         if units < 0:
#             raise ValueError(f'Received an invalid value for units, expected '
#                              f'a positive integer, got {units}.')
#         super(GRUICell, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activations.get(activation)
#         self.recurrent_activation = activations.get(recurrent_activation)
#         self.use_bias = use_bias
#
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.recurrent_initializer = initializers.get(recurrent_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#
#         self.dropout = min(1., max(0., dropout))
#         self.recurrent_dropout = min(1., max(0., recurrent_dropout))
#
#         implementation = kwargs.pop('implementation', 1)
#         if self.recurrent_dropout != 0 and implementation != 1:
#             logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
#             self.implementation = 1
#         else:
#             self.implementation = implementation
#         self.reset_after = reset_after
#
#     @property
#     def state_size(self):
#         return self.units
#
#     def build(self, input_shape):
#         if self.units == None:
#             self.units = input_shape[-1]
#         input_dim = input_shape[-1]
#         super(GRUICell, self).build(input_shape)
#
#         self.t_kerknel = self.add_weight(
#             shape=(1, input_dim),
#             name='time_kernel',
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#         )
#         self.t_bias = self.add_weight(
#             shape=(1, input_dim),
#             name='time_bias',
#             initializer=self.bias_initializer,
#             regularizer=self.bias_regularizer,
#             constraint=self.bias_constraint,
#         )
#         self.built = True
#
#     def time_map(self, x):
#         tf.nn.relu()
#         return time
#
#     def call(self, inputs, states, training=None, **kwargs):
#         # Reset Gate (r) # Update Gate (z) # Candidate (h)
#         states = states[0] if tf.nest.is_nested(states) else states
#         self.time = kwargs.pop('time', None)
#         if self.time == None:
#             logging.warning(
#                 '[!] There is no "Time map" in kwargs.'
#             )
#             h_tm1 = states
#         else:
#             time = self.t_kerknel * self.time + self.t_bias
#             h_tm1 = states * time
#
#         return super(GRUICell, self).call(
#             inputs, mask, training=training)
#
#     def get_config(self):
#         config = super(GRUICell, self).get_config()
#         config.update({
#             't_kernel' : self.t_kerknel,
#             't_bais' : self.t_bias,
#         })
#         return config


########################################################################################################################

########################################################################################################################
class GRUI(recurrent.GRU):
    """
    Multivariate Time Series Imputation with Generative Adversarial Networks
    https://papers.nips.cc/paper/2018/hash/96b9bff013acedfb1d140579e2fbeb63-Abstract.html
    """
    def __init__(self, units, **kwargs):
        super(GRUI, self).__init__(units, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.t_kerknel = self.add_weight(
            shape=(1, input_dim),
            name='time_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.t_bias = self.add_weight(
            shape=(1, input_dim),
            name='time_bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

    def get_time_map(self, time):
        x = tf.nn.relu(self.t_kerknel * time + self.t_bias)
        time_map = 1 / tf.math.exp(x)
        return time_map

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        time = kwargs.pop('time', None)
        if time == None:
            logging.warning(
                '[!] There is no "Time map" in kwargs.'
            )
        else:
            time_map = self.get_time_map(time)
            inputs *= time_map
        return super(GRUI, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    def get_config(self):
        config = super(GRUI, self).get_config()
        config.update({
            't_kernel' : self.t_kerknel,
            't_bais' : self.t_bias,
        })
        return config

########################################################################################################################

