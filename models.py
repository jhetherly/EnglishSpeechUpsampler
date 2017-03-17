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


def build_1d_conv_layer(prev_tensor, prev_conv_depth,
                        conv_window, conv_depth,
                        act, layer_number,
                        tensorboard_output=False,
                        name=None):
    with tf.name_scope('{}_layer_weights'.format(layer_number)):
        W = weight_variable([conv_window,
                             prev_conv_depth,
                             conv_depth])
        if tensorboard_output:
            histogram_variable_summaries(W)
    with tf.name_scope('{}_layer_biases'.format(layer_number)):
        b = bias_variable([conv_depth])
        if tensorboard_output:
            histogram_variable_summaries(b)
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = conv1d(prev_tensor, W) + b
        if tensorboard_output:
            histogram_variable_summaries(conv)
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        h = act(conv, name=name)
        if tensorboard_output:
            histogram_variable_summaries(h)
    return h


def build_1d_conv_layer_with_res(prev_tensor, prev_conv_depth,
                                 conv_window, conv_depth,
                                 res, act, layer_number,
                                 tensorboard_output=False,
                                 name=None):
    with tf.name_scope('{}_layer_weights'.format(layer_number)):
        W = weight_variable([conv_window,
                             prev_conv_depth,
                             conv_depth])
        if tensorboard_output:
            histogram_variable_summaries(W)
    with tf.name_scope('{}_layer_biases'.format(layer_number)):
        b = bias_variable([conv_depth])
        if tensorboard_output:
            histogram_variable_summaries(b)
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = conv1d(prev_tensor, W) + b
        if tensorboard_output:
            histogram_variable_summaries(conv)
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        h = act(tf.add(conv, res), name=name)
        if tensorboard_output:
            histogram_variable_summaries(h)
    return h

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
            y = tf.identity(preact, name=scope_name)
            if tensorboard_output:
                histogram_variable_summaries(y)

        return x, y


def three_layer_conv_model(input_type, input_shape,
                           first_conv_window=30, first_conv_depth=128,
                           second_conv_window=10, second_conv_depth=64,
                           third_conv_window=15,
                           tensorboard_output=False,
                           scope_name='3-layer_conv'):

    with tf.name_scope(scope_name):
        # input of the model (examples)
        s = [None]
        for i in input_shape:
            s.append(i)
        x = tf.placeholder(input_type, shape=s)

        # first conv layer
        h1 = build_1d_conv_layer(x, 1,
                                 first_conv_window, first_conv_depth,
                                 tf.nn.elu, 1,
                                 tensorboard_output)

        # second conv layer
        h2 = build_1d_conv_layer(h1, first_conv_depth,
                                 second_conv_window, second_conv_depth,
                                 tf.nn.elu, 2,
                                 tensorboard_output)

        # third (last) conv layer
        y = build_1d_conv_layer(h2, second_conv_depth,
                                third_conv_window, 1,
                                tf.identity, 3,
                                tensorboard_output,
                                scope_name)

        return x, y


def three_layer_conv_with_res_model(input_type, input_shape,
                                    first_conv_window=30, first_conv_depth=128,
                                    second_conv_window=10,
                                    second_conv_depth=64,
                                    third_conv_window=15,
                                    tensorboard_output=False,
                                    scope_name='3-layer_conv_res'):

    with tf.name_scope(scope_name):
        # input of the model (examples)
        s = [None]
        for i in input_shape:
            s.append(i)
        x = tf.placeholder(input_type, shape=s)

        # first conv layer
        h1 = build_1d_conv_layer(x, 1,
                                 first_conv_window, first_conv_depth,
                                 tf.nn.elu, 1,
                                 tensorboard_output)

        # second conv layer
        h2 = build_1d_conv_layer(h1, first_conv_depth,
                                 second_conv_window, second_conv_depth,
                                 tf.nn.elu, 2,
                                 tensorboard_output)

        # third (last) conv layer
        y = build_1d_conv_layer_with_res(h2, second_conv_depth,
                                         third_conv_window, 1,
                                         x, tf.identity, 3,
                                         tensorboard_output,
                                         scope_name)

        return x, y


def five_layer_conv_model(input_type, input_shape,
                          first_conv_window=30, first_conv_depth=256,
                          second_conv_window=20, second_conv_depth=128,
                          third_conv_window=10, third_conv_depth=64,
                          fourth_conv_window=5, fourth_conv_depth=32,
                          fifth_conv_window=5,
                          tensorboard_output=False,
                          scope_name='5-layer_conv'):

    with tf.name_scope(scope_name):
        # input of the model (examples)
        s = [None]
        for i in input_shape:
            s.append(i)
        x = tf.placeholder(input_type, shape=s)

        # first conv layer
        h1 = build_1d_conv_layer(x, 1,
                                 first_conv_window, first_conv_depth,
                                 tf.nn.elu, 1,
                                 tensorboard_output)

        # second conv layer
        h2 = build_1d_conv_layer(h1, first_conv_depth,
                                 second_conv_window, second_conv_depth,
                                 tf.nn.elu, 2,
                                 tensorboard_output)

        # third conv layer
        h3 = build_1d_conv_layer(h2, second_conv_depth,
                                 third_conv_window, third_conv_depth,
                                 tf.nn.elu, 3,
                                 tensorboard_output)

        # fourth conv layer
        h4 = build_1d_conv_layer(h3, third_conv_depth,
                                 fourth_conv_window, fourth_conv_depth,
                                 tf.nn.elu, 4,
                                 tensorboard_output)

        # fifth (last) conv layer
        y = build_1d_conv_layer(h4, fourth_conv_depth,
                                fifth_conv_window, 1,
                                tf.identity, 5,
                                tensorboard_output,
                                scope_name)

        return x, y

# #################
# #################
