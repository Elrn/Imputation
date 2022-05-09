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
class GRUI_test(recurrent.RNN):
    def __init__(self, units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
        cell = recurrent.GRUCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
        )
        super(GRUI_test, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs
        )

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
        self.time = kwargs.pop('time', None)
        if self.time == None:
            logging.warning(
                '[!] There is no "Time map" in kwargs.'
            )
        else:
            time_map = self.get_time_map(self.time)
            inputs *= time_map
        return super(GRUI_test, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation,
            'reset_after':
                self.reset_after
        }
        config.update({
            't_kernel': self.t_kerknel,
            't_bais': self.t_bias,
        })
        # config.update(_config_for_enable_caching_device(self.cell))
        base_config = super(GRUI_test, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    # @classmethod
    # def from_config(cls, config):
    #     if 'implementation' in config and config['implementation'] == 0:
    #         config['implementation'] = 1
    #     return cls(**config)

    # def get_config(self):
    #     config = super(GRUI_test, self).get_config()
    #     config.update({
    #         't_kernel' : self.t_kerknel,
    #         't_bais' : self.t_bias,
    #     })
    #     return config

########################################################################################################################

