import tensorflow as tf

def imputation(y, y_, mask, fake_logit, lambda_=1):
    """
    Multivariate Time Series Imputation with Generative Adversarial Networks
    """
    def reconstruction(y, y_, mask):
        y, y_ = y * mask, y_ * mask
        return tf.norm(y - y_, 2)

    def discriminative(fake_logit):
        return -tf.reduce_mean(fake_logit)

    return reconstruction(y, y_, mask) + lambda_* discriminative(fake_logit)

def discriminator(real_logit, fake_logit):
    real_loss = tf.reduce_mean(real_logit)
    fake_loss = tf.reduce_mean(fake_logit)
    return fake_loss - real_loss

def generator(fake_logit):
    return -tf.reduce_mean(fake_logit)