import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import flags
FLAGS = flags.FLAGS
import losses
import layers
from data import _utils as dutils

########################################################################################################################
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.D = discriminator
        self.G = generator
        self.time_map_generator = dutils.time_lag_matrix()

    def compile(self, d_opt, g_opt):
        super(GAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.d_loss = losses.discriminator
        self.g_loss = losses.imputation

    def gradient_penalty(self, real, fake, time):
        alpha = tf.random.normal([FLAGS.bsz, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.D((interpolated, time), training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def regularizer(self, x, time):
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = self.D((x, time), training=True)
        grads = tape.gradient(logits, x)[0]
        norm = tf.norm(grads, 2)
        return 10. * norm

    @tf.function
    def train_step(self, inputs):
        data, time = inputs
        # time_lag = self.time_map_generator(time)
        time_lag = tf.zeros([FLAGS.bsz, FLAGS.row_lengths, FLAGS.input_dim])
        x = (data, time)
        with tf.GradientTape() as tape:
            fake = self.G(x, training=True)
            real_logits = self.D(x, training=True)
            fake_logits = self.D((fake, time_lag), training=True)
            d_loss = self.d_loss(real_logits, fake_logits)
            gp = self.gradient_penalty(data, fake, time_lag)
            d_loss = d_loss + gp * 10.
        d_grad = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(
            zip(d_grad, self.D.trainable_variables)
        )
        with tf.GradientTape() as tape:
            fake = self.G(x, training=True)
            fake_logits = self.D((fake, time_lag), training=True)
            mask = tf.math.logical_not(tf.math.is_nan(x))
            mask = tf.cast(mask, tf.float32)
            g_loss = self.g_loss(x, fake, mask, fake_logits, FLAGS.lambda_)
        g_grad = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(
            zip(g_grad, self.G.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss, "GP": gp, 'real':real_logits, 'fake':fake_logits}

    # def call(self, inputs):
    #     return self.G(inputs)

########################################################################################################################
def build(input_shape, model):
    inputs = tf.keras.layers.Input(shape=input_shape)
    time = tf.keras.layers.Input(shape=input_shape)
    inputs = [inputs, time]
    output = model(inputs)
    model = tf.keras.Model(inputs, output, name=None)
    return model

def discriminator(units, drop_rate=0.2):
    def main(x):
        x, time = x
        # x = Embedding(input_dim=1000, output_dim=64)(x)
        x = Masking(mask_value=np.nan)(x)
        x = layers.GRUI(
            units
        )(x, time=time)
        x = Dropout(drop_rate)(x)
        x = Dense(1)(x)
        return x
    return main

def generator(units, drop_rate=0.2):
    """
    입력 데이터에 의해 생성된 노이즈(z)를 입력으로 Inputation을 진행

    :return:
    """
    def main(x):
        x, time = x
        x = Masking(mask_value=np.nan)(x)
        z = z_generator(FLAGS.input_dim)((x, time))
        fake = layers.GRUI(
            units=FLAGS.input_dim,
            return_sequences=True,
            # return_state=True,
        )(z, time=time)
        return fake
    return main

def z_generator(units):
    """
    단순히 Random Noise를 활용하기 보다, Inputation 대상에 맞는 z를 생성하기 위함

    1. return_sequences 를 사용해 같은 길이를 추출?
    2. Dense를 사용해 Non-linearity 추가?
    :return: z
    """
    def main(x):
        x, time = x
        z = layers.GRUI(
            units=units,
            return_sequences=True
        )(x, time=time)
        return z
    return main


########################################################################################################################