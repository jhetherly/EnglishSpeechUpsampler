import tensorflow as tf


def mse(sname, true, model):
    with tf.name_scope(sname):
        waveform_loss = tf.reduce_mean(tf.square(tf.subtract(true, model)))
    tf.summary.scalar(sname, waveform_loss)
    return waveform_loss


def l2(sname, true, model):
    with tf.name_scope(sname):
        waveform_loss = tf.nn.l2_loss(tf.subtract(true, model))
    tf.summary.scalar(sname, waveform_loss)
    return waveform_loss


def linf(sname, true, model):
    with tf.name_scope(sname):
        waveform_loss = tf.reduce_max(tf.abs(tf.subtract(true, model)))
    tf.summary.scalar(sname, waveform_loss)
    return waveform_loss


def geo_mean(sname, true, model):
    with tf.name_scope(sname):
        waveform_loss = tf.exp(tf.reduce_mean(tf.log1p(
                                tf.abs(tf.subtract(true, model)))))
    tf.summary.scalar(sname, waveform_loss)
    return waveform_loss
