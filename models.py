import numpy as np
import tensorflow as tf


# ###################
# TENSORBOARD HELPERS
# ###################

def comprehensive_variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def histogram_variable_summaries(var):
    """
    Attach a histogram summary to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        tf.summary.histogram('histogram', var)

# ###################
# ###################


# ######################
# LAYER HELPER FUNCTIONS
# ######################

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv1d(x, W, name=None):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME', name=name)

# ######################
# ######################


# #################
# MODEL DEFINITIONS
# #################

def single_fully_connected_model(input_type, input_shape,
                                 n_inputs, n_weights,
                                 tensorboard_output=True,
                                 scope_name='single_fully_connected_layer'):

    with tf.name_scope(scope_name):
        # input of the model (examples)
        s = [None]
        shape_prod = 1
        for i in input_shape:
            s.append(i)
            shape_prod *= i
        x = tf.placeholder(input_type, shape=s)
        x_ = tf.reshape(x, [-1, shape_prod])

        # first conv layer
        with tf.name_scope('first_layer_weights'):
            s = []
            s.append(shape_prod)
            s.append(n_weights)
            # W = tf.Variable(initial_value=np.eye(shape_prod, n_weights).astype(np.float32))
            W = weight_variable(s)
            if tensorboard_output:
                histogram_variable_summaries(W)
        with tf.name_scope('first_layer_biases'):
            b = bias_variable([n_weights])
            if tensorboard_output:
                histogram_variable_summaries(b)
        with tf.name_scope('first_layer_preactivation'):
            preact = tf.matmul(x_, W) + b
            if tensorboard_output:
                histogram_variable_summaries(preact)
        with tf.name_scope('first_layer_activation'):
            y = tf.identity(preact)
            if tensorboard_output:
                histogram_variable_summaries(y)

        return x, y


def three_layer_conv_model(input_type, input_shape,
                           first_conv_window=30, first_conv_depth=128,
                           second_conv_window=10, second_conv_depth=64,
                           thrid_conv_window=15,
                           tensorboard_output=True, scope_name='3-layer_conv'):

    with tf.name_scope(scope_name):
        # input of the model (examples)
        s = [None]
        for i in input_shape:
            s.append(i)
        x = tf.placeholder(input_type, shape=s)

        # first conv layer
        with tf.name_scope('first_layer_weights'):
            W_conv1 = weight_variable([first_conv_window,
                                       1,
                                       first_conv_depth])
            if tensorboard_output:
                histogram_variable_summaries(W_conv1)
        with tf.name_scope('first_layer_biases'):
            b_conv1 = bias_variable([first_conv_depth])
            if tensorboard_output:
                histogram_variable_summaries(b_conv1)
        with tf.name_scope('first_layer_conv_preactivation'):
            conv1 = conv1d(x, W_conv1) + b_conv1
            if tensorboard_output:
                histogram_variable_summaries(conv1)
        with tf.name_scope('first_layer_conv_activation'):
            h_conv1 = tf.nn.elu(conv1)
            if tensorboard_output:
                histogram_variable_summaries(h_conv1)

        # second conv layer
        with tf.name_scope('second_layer_weights'):
            W_conv2 = weight_variable([second_conv_window,
                                       first_conv_depth,
                                       second_conv_depth])
            if tensorboard_output:
                histogram_variable_summaries(W_conv2)
        with tf.name_scope('second_layer_biases'):
            b_conv2 = bias_variable([second_conv_depth])
            if tensorboard_output:
                histogram_variable_summaries(b_conv2)
        with tf.name_scope('second_layer_conv_preactivation'):
            conv2 = conv1d(h_conv1, W_conv2) + b_conv2
            if tensorboard_output:
                histogram_variable_summaries(conv2)
        with tf.name_scope('second_layer_conv_activation'):
            h_conv2 = tf.nn.elu(conv2)
            if tensorboard_output:
                histogram_variable_summaries(h_conv2)

        # third (last) conv layer
        with tf.name_scope('third_layer_weights'):
            W_conv3 = weight_variable([thrid_conv_window,
                                       second_conv_depth,
                                       1])
            if tensorboard_output:
                histogram_variable_summaries(W_conv3)
        with tf.name_scope('third_layer_biases'):
            b_conv3 = bias_variable([1])
            if tensorboard_output:
                histogram_variable_summaries(b_conv3)
        with tf.name_scope('third_layer_conv_preactivation'):
            conv3 = conv1d(h_conv2, W_conv3) + b_conv3
            if tensorboard_output:
                histogram_variable_summaries(conv3)
        with tf.name_scope('third_layer_conv_activation'):
            # essentially a linear activation
            y_conv = tf.identity(conv3)
            if tensorboard_output:
                comprehensive_variable_summaries(y_conv)

        return x, y_conv

# #################
# #################
