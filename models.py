import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import flags
FLAGS = flags.FLAGS
import losses

########################################################################################################################
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.D = discriminator
        self.G = generator

    def compile(self, d_opt, g_opt):
        super(GAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.d_loss = losses.discriminator
        self.g_loss = losses.imputation

    def gradient_penalty(self, real, fake):
        bsz = real.shape[0]
        alpha = tf.random.normal([bsz, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.D(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            fake = self.G(inputs, training=True)
            real_logits = self.D(inputs, training=True)
            fake_logits = self.D(fake, training=True)
            d_cost = self.d_loss(real_logits, fake_logits)
            gp = self.gradient_penalty(inputs, fake)
            d_loss = d_cost + gp * self.gp_weight
        d_grad = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(
            zip(d_grad, self.D.trainable_variables)
        )

        with tf.GradientTape() as tape:
            fake = self.G(inputs, training=True)
            fake_logits = self.D(fake, training=True)
            mask = tf.math.logical_not(tf.math.is_nan(inputs))
            g_loss = self.g_loss(inputs, fake, mask, fake_logits, FLAGS.lambda_)
        g_grad = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(
            zip(g_grad, self.G.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss}
    #
    # def call(self, inputs, training=None):
    #     return

########################################################################################################################
def build(input_shape, model):
    input = tf.keras.layers.Input(shape=input_shape)
    output = model(input)
    model = tf.keras.Model(input, output, name=None)
    return model

def discriminator(units=64, drop_rate=0.2):
    def main(x):
        # x = Embedding(input_dim=1000, output_dim=64)(x)
        x = tf.keras.layers.RNN(
            tf.keras.layers.GRUICell(units),
            return_sequences=True,
            return_state=True
        )(x)
        x = Dropout(drop_rate)(x)
        x = Dense(1)(x)
        return x
    return main

def generator(units=64, drop_rate=0.2):
    def main(x):
        z = z_generator()(x)
        x = tf.keras.layers.RNN(
            tf.keras.layers.GRUICell(units, return_sequences=True),
            return_sequences=True,
            return_state=True
        )(z)
        x = Dropout(drop_rate)(x)
        x = Dense(FLAGS.input_dims)(x)
        return x, z
    return main

def z_generator():
    def main(x):
        x = GRU(units=x.shape[0], return_sequences=True)(x)
        return x
    return main

########################################################################################################################