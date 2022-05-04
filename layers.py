import tensorflow as tf
from tensorflow.keras.layers import *
from keras import activations
from keras import backend
from keras.engine import base_layer
from keras.layers import recurrent

from tensorflow.python.platform import tf_logging as logging
from keras.layers.recurrent import *

########################################################################################################################
# Dropout 추가 고려
class GRUICell(AbstractRNNCell):
    """
    https://yjjo.tistory.com/18

    """
    def __init__(self,
                 units=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 reset_after=True,
                 **kwargs
                 ):
        if units < 0:
            raise ValueError(f'Received an invalid value for units, expected '
                             f'a positive integer, got {units}.')
        super(GRUICell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        self.reset_after = reset_after

    def build(self, input_shape):
        if self.unit == None:
            self.unit = input_shape[-1]
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.r_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.bias = self.add_weight(
            shape=(2, 3 * self.units),
            name='bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
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
        self.built = True

    def call(self, inputs, states, time):
        # Reset Gate (r)
        # Update Gate (z)
        # Candidate (h)
        bias, r_bias = tf.unstack(self.bias)
        time = self.t_kerknel * time + self.t_bias
        h_tm1 = states[0] * time

        x_z = backend.dot(inputs, self.kernel[:, :self.units])
        x_r = backend.dot(inputs, self.kernel[:, self.units:self.units * 2])
        x_h = backend.dot(inputs, self.kernel[:, self.units * 2:])
        x_z = backend.bias_add(x_z, bias[:self.units])
        x_r = backend.bias_add(x_r, bias[self.units: self.units * 2])
        x_h = backend.bias_add(x_h, bias[self.units * 2:])

        r_z = backend.dot(h_tm1, self.r_kernel[:, :self.units])
        r_r = backend.dot(h_tm1, self.r_kernel[:, self.units:self.units * 2])
        r_h = backend.dot(h_tm1, self.r_kernel[:, self.units * 2:])
        r_z = backend.bias_add(r_z, r_bias[:self.units])
        r_r = backend.bias_add(r_r, r_bias[self.units: self.units * 2])
        r_h = backend.bias_add(r_h, r_bias[self.units * 2:])

        z = self.recurrent_activation(x_z + r_z)
        r = self.recurrent_activation(x_r + r_r)

        r_h *= r # reset
        h = self.activation(x_h + r_h)
        h_t = z * h_tm1 + (1 - z) * h
        new_state = [h_t] if tf.nest.is_nested(states) else h_t

        return h_t, new_state

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'implementation': self.implementation,
            'reset_after': self.reset_after
        }
        config.update({
            't_kernel' : self.t_kerknel,
            't_bais' : self.t_bias,
        })
        base_config = super(GRUICell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


########################################################################################################################
# class GRUCell(DropoutRNNCellMixin, Layer):
#     def __init__(self,
#                  units,
#                  activation='tanh',
#                  recurrent_activation='hard_sigmoid',
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  recurrent_regularizer=None,
#                  bias_regularizer=None,
#                  kernel_constraint=None,
#                  recurrent_constraint=None,
#                  bias_constraint=None,
#                  dropout=0.,
#                  recurrent_dropout=0.,
#                  reset_after=True,
#                  **kwargs):
#         if units < 0:
#             raise ValueError(f'Received an invalid value for units, expected '
#                              f'a positive integer, got {units}.')
#         # By default use cached variable under v2 mode, see b/143699808.
#         if ops.executing_eagerly_outside_functions():
#             self._enable_caching_device = kwargs.pop('enable_caching_device', True)
#         else:
#             self._enable_caching_device = kwargs.pop('enable_caching_device', False)
#         super(GRUCell, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activations.get(activation)
#         self.recurrent_activation = activations.get(recurrent_activation)
#         self.use_bias = use_bias
#
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.recurrent_initializer = initializers.get(recurrent_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.recurrent_constraint = constraints.get(recurrent_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
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
#         self.state_size = self.units
#         self.output_size = self.units
#
#     @tf_utils.shape_type_conversion
#     def build(self, input_shape):
#         input_dim = input_shape[-1]
#         self.kernel = self.add_weight(
#             shape=(input_dim, self.units * 3),
#             name='kernel',
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#         )
#         self.recurrent_kernel = self.add_weight(
#             shape=(self.units, self.units * 3),
#             name='recurrent_kernel',
#             initializer=self.recurrent_initializer,
#             regularizer=self.recurrent_regularizer,
#             constraint=self.recurrent_constraint,
#         )
#         if self.use_bias:
#             if not self.reset_after:
#                 bias_shape = (3 * self.units,)
#             else:
#                 # separate biases for input and recurrent kernels
#                 # Note: the shape is intentionally different from CuDNNGRU biases
#                 # `(2 * 3 * self.units,)`, so that we can distinguish the classes
#                 # when loading and converting saved weights.
#                 bias_shape = (2, 3 * self.units)
#             self.bias = self.add_weight(shape=bias_shape,
#                                         name='bias',
#                                         initializer=self.bias_initializer,
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint,
#                                         caching_device=default_caching_device)
#         else:
#             self.bias = None
#         self.built = True
#
#     def call(self, inputs, states, training=None):
#         h_tm1 = states[0] if tf.nest.is_nested(states) else states  # previous memory
#
#         dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
#         rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
#             h_tm1, training, count=3)
#
#         if self.use_bias:
#             if not self.reset_after:
#                 input_bias, recurrent_bias = self.bias, None
#             else:
#                 input_bias, recurrent_bias = tf.unstack(self.bias)
#
#         if self.implementation == 1:
#             if 0. < self.dropout < 1.:
#                 inputs_z = inputs * dp_mask[0]
#                 inputs_r = inputs * dp_mask[1]
#                 inputs_h = inputs * dp_mask[2]
#             else:
#                 inputs_z = inputs
#                 inputs_r = inputs
#                 inputs_h = inputs
#             x_z = backend.dot(inputs_z, self.kernel[:, :self.units])
#             x_r = backend.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
#             x_h = backend.dot(inputs_h, self.kernel[:, self.units * 2:])
#
#             if self.use_bias:
#                 x_z = backend.bias_add(x_z, input_bias[:self.units])
#                 x_r = backend.bias_add(x_r, input_bias[self.units: self.units * 2])
#                 x_h = backend.bias_add(x_h, input_bias[self.units * 2:])
#             if 0. < self.recurrent_dropout < 1.:
#                 h_tm1_z = h_tm1 * rec_dp_mask[0]
#                 h_tm1_r = h_tm1 * rec_dp_mask[1]
#                 h_tm1_h = h_tm1 * rec_dp_mask[2]
#             else:
#                 h_tm1_z = h_tm1
#                 h_tm1_r = h_tm1
#                 h_tm1_h = h_tm1
#
#             recurrent_z = backend.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
#             recurrent_r = backend.dot(h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
#             if self.reset_after and self.use_bias:
#                 recurrent_z = backend.bias_add(recurrent_z, recurrent_bias[:self.units])
#                 recurrent_r = backend.bias_add(recurrent_r, recurrent_bias[self.units:self.units * 2])
#
#             z = self.recurrent_activation(x_z + recurrent_z)
#             r = self.recurrent_activation(x_r + recurrent_r)
#             # reset gate applied after/before matrix multiplication
#             if self.reset_after:
#                 recurrent_h = backend.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
#                 if self.use_bias:
#                     recurrent_h = backend.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
#                 recurrent_h = r * recurrent_h
#             else:
#                 recurrent_h = backend.dot(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
#
#             hh = self.activation(x_h + recurrent_h)
#         else:
#             if 0. < self.dropout < 1.:
#                 inputs = inputs * dp_mask[0]
#
#             # inputs projected by all gate matrices at once
#             matrix_x = backend.dot(inputs, self.kernel)
#             if self.use_bias:
#                 # biases: bias_z_i, bias_r_i, bias_h_i
#                 matrix_x = backend.bias_add(matrix_x, input_bias)
#
#             x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)
#
#             if self.reset_after:
#                 # hidden state projected by all gate matrices at once
#                 matrix_inner = backend.dot(h_tm1, self.recurrent_kernel)
#                 if self.use_bias:
#                     matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)
#             else:
#                 # hidden state projected separately for update/reset and new
#                 matrix_inner = backend.dot(
#                     h_tm1, self.recurrent_kernel[:, :2 * self.units])
#
#             recurrent_z, recurrent_r, recurrent_h = array_ops.split(
#                 matrix_inner, [self.units, self.units, -1], axis=-1)
#
#             z = self.recurrent_activation(x_z + recurrent_z)
#             r = self.recurrent_activation(x_r + recurrent_r)
#
#             if self.reset_after:
#                 recurrent_h = r * recurrent_h
#             else:
#                 recurrent_h = backend.dot(
#                     r * h_tm1, self.recurrent_kernel[:, 2 * self.units:])
#
#             hh = self.activation(x_h + recurrent_h)
#         # previous and candidate state mixed by update gate
#         h = z * h_tm1 + (1 - z) * hh
#         new_state = [h] if tf.nest.is_nested(states) else h
#         return h, new_state
#
#     def get_config(self):
#         config = {
#             'units': self.units,
#             'activation': activations.serialize(self.activation),
#             'recurrent_activation':
#                 activations.serialize(self.recurrent_activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'recurrent_initializer':
#                 initializers.serialize(self.recurrent_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'recurrent_regularizer':
#                 regularizers.serialize(self.recurrent_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'recurrent_constraint':
#                 constraints.serialize(self.recurrent_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint),
#             'dropout': self.dropout,
#             'recurrent_dropout': self.recurrent_dropout,
#             'implementation': self.implementation,
#             'reset_after': self.reset_after
#         }
#         # config.update(_config_for_enable_caching_device(self))
#         base_config = super(GRUCell, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
#         return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)