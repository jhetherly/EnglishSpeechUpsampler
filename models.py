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

def subpixel_reshuffle_1D_impl(X, m):
    """
    performs a 1-D subpixel reshuffle of the input 2-D tensor
    assumes the last dimension of X is the filter dimension
    ref: https://github.com/Tetrachrome/subpixel
    """
    return tf.transpose(tf.stack([tf.reshape(x, (-1,)) for x
                                  in tf.split(X, m, axis=1)]))


def subpixel_reshuffle_1D(X, m, name=None):
    """
    maps over the batch dimension
    """
    return tf.map_fn(lambda x: subpixel_reshuffle_1D_impl(x, m), X, name=name)


def subpixel_restack_impl(X, n_prime, m_prime, name=None):
    """
    performs a subpixel restacking such that it restacks columns of a 2-D
    tensor onto the rows
    """
    bsize = tf.shape(X)[0]
    r_n = n_prime - X.get_shape().as_list()[1]
    total_new_space = r_n*m_prime
    to_stack = tf.slice(X, [0, 0, m_prime], [-1, -1, -1])
    to_stack = tf.slice(tf.reshape(to_stack, (bsize, -1)),
                        [0, 0], [-1, total_new_space])
    to_stack = tf.reshape(to_stack, (bsize, -1, m_prime))
    to_stack = tf.slice(to_stack, [0, 0, 0], [-1, r_n, -1])
    return tf.concat((tf.slice(X, [0, 0, 0], [-1, -1, m_prime]), to_stack),
                     axis=1, name=name)


def subpixel_restack(X, n_prime, m_prime=None, name=None):
    n = X.get_shape().as_list()[1]
    m = X.get_shape().as_list()[2]
    r_n = n_prime - n
    if m_prime is None:
        for i in range(1, m):
            r_m = i
            m_prime = m - r_m
            if r_m*n >= m_prime*r_n:
                break
    return subpixel_restack_impl(X, n_prime, m_prime, name=name)


def BatchNorm(T, is_training, scope):
    # tf.cond takes nullary functions as its first and second arguments
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(T,
                            decay=0.99,
                            # zero_debias_moving_mean=True,
                            is_training=is_training,
                            center=True, scale=True,
                            updates_collections=None,
                            scope=scope,
                            reuse=False),
                   lambda: tf.contrib.layers.batch_norm(T,
                            decay=0.99,
                            is_training=is_training,
                            center=True, scale=True,
                            updates_collections=None,
                            scope=scope,
                            reuse=True))


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv1d(x, W, stride=1, padding='SAME', name=None):
    return tf.nn.conv1d(x, W, stride=stride, padding=padding, name=name)


def build_1d_conv_layer(prev_tensor, prev_conv_depth,
                        conv_window, conv_depth,
                        act, layer_number,
                        stride=1,
                        padding='SAME',
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
        conv = conv1d(prev_tensor, W, stride=stride, padding=padding) + b
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


def build_downsampling_block(input_tensor,
                             filter_size, stride,
                             layer_number,
                             act=tf.nn.relu,
                             is_training=True,
                             depth=None,
                             padding='VALID',
                             tensorboard_output=False,
                             name=None):

    # assume this layer is twice the depth of the previous layer if no depth
    # information is given
    if depth is None:
        depth = 2*input_tensor.get_shape().as_list()[-1]

    with tf.name_scope('{}_layer_weights'.format(layer_number)):
        W = weight_variable([filter_size,
                             input_tensor.get_shape().as_list()[-1],
                             depth])
        if tensorboard_output:
            histogram_variable_summaries(W)
    with tf.name_scope('{}_layer_biases'.format(layer_number)):
        b = bias_variable([depth])
        if tensorboard_output:
            histogram_variable_summaries(b)
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        l = tf.nn.conv1d(input_tensor, W, stride=stride,
                         padding=padding, name=name) + b
        if tensorboard_output:
            histogram_variable_summaries(l)
    with tf.name_scope('{}_layer_batch_norm'.format(layer_number)) as scope:
        # l = tf.nn.dropout(l, keep_prob=0.25)
        l = BatchNorm(l, is_training, scope)
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        l = act(l, name=name)
        if tensorboard_output:
            histogram_variable_summaries(l)

    return l


def build_upsampling_block(input_tensor, residual_tensor,
                           filter_size,
                           layer_number,
                           act=tf.nn.relu,
                           is_training=True,
                           depth=None,
                           padding='VALID',
                           tensorboard_output=False,
                           name=None):

    # assume this layer is half the depth of the previous layer if no depth
    # information is given
    if depth is None:
        depth = int(input_tensor.get_shape().as_list()[-1]/2)

    with tf.name_scope('{}_layer_weights'.format(layer_number)):
        W = weight_variable([filter_size,
                             input_tensor.get_shape().as_list()[-1],
                             depth])
        if tensorboard_output:
            histogram_variable_summaries(W)
    with tf.name_scope('{}_layer_biases'.format(layer_number)):
        b = bias_variable([depth])
        if tensorboard_output:
            histogram_variable_summaries(b)
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        l = tf.nn.conv1d(input_tensor, W, stride=1,
                         padding=padding, name=name) + b
        if tensorboard_output:
            histogram_variable_summaries(l)
    with tf.name_scope('{}_layer_batch_norm'.format(layer_number)) as scope:
        # l = tf.nn.dropout(l, keep_prob=0.25)
        l = BatchNorm(l, is_training, scope)
        # l = tf.nn.l2_normalize(l, dim=2)
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        l = act(l, name=name)
        if tensorboard_output:
            histogram_variable_summaries(l)
    with tf.name_scope('{}_layer_subpixel_reshuffle'.format(layer_number)):
        l = subpixel_reshuffle_1D(l,
                                  residual_tensor.get_shape().as_list()[-1],
                                  name=name)
        if tensorboard_output:
            histogram_variable_summaries(l)
        # print('after residual_tensor: {}'.format(
        #     residual_tensor.get_shape().as_list()[1:]))
        # print('after subpixel_reshuffle: {}'.format(
        #     l.get_shape().as_list()[1:]))
    with tf.name_scope('{}_layer_stacking'.format(layer_number)):
        sliced = tf.slice(residual_tensor,
                          begin=[0, 0, 0],
                          size=[-1, l.get_shape().as_list()[1], -1])
        l = tf.concat((l, sliced), axis=2, name=name)
        if tensorboard_output:
            histogram_variable_summaries(l)
        # print('sliced: {}'.format(sliced.get_shape().as_list()[1:]))
        # print('after concat: {}'.format(l.get_shape().as_list()[1:]))

    return l
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
            # W = tf.Variable(initial_value=np.eye(shape_prod,
            #                                      n_weights).astype(
            #                                      np.float32))
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

        # # second conv layer
        # h2 = build_1d_conv_layer(h1, first_conv_depth,
        # # h2 = build_1d_conv_layer_with_res(h1, first_conv_depth,
        #                          second_conv_window, second_conv_depth,
        #                         #  h1, tf.nn.elu, 2,
        #                          tf.nn.elu, 2,
        #                          tensorboard_output)

        # third (last) conv layer
        # y = build_1d_conv_layer_with_res(h2, second_conv_depth,
        y = build_1d_conv_layer_with_res(h1, first_conv_depth,
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


def deep_residual_network(input_type, input_shape,
                          number_of_downsample_layers=8,
                          channel_multiple=8,
                          initial_filter_window=5,
                          initial_stride=2,
                          downsample_filter_window=3,
                          downsample_stride=2,
                          bottleneck_filter_window=4,
                          bottleneck_stride=2,
                          upsample_filter_window=3,
                          tensorboard_output=False,
                          scope_name='deep_residual'):

    print('layer summary for {} network'.format(scope_name))
    downsample_layers = []
    upsample_layers = []

    # tf.reset_default_graph()
    with tf.name_scope(scope_name):
        # training flag
        train_flag = tf.placeholder(tf.bool)

        # input of the model (examples)
        s = [None]
        for i in input_shape:
            s.append(i)
        x = tf.placeholder(input_type, shape=s)
        input_size = s[-2]
        num_of_channels = s[-1]
        print('input: {}'.format(x.get_shape().as_list()[1:]))

        d1 = build_downsampling_block(x,
                                      filter_size=initial_filter_window,
                                      stride=initial_stride,
                                      tensorboard_output=tensorboard_output,
                                      depth=channel_multiple*num_of_channels,
                                      is_training=train_flag,
                                      layer_number=1)
        print('downsample layer: {}'.format(d1.get_shape().as_list()[1:]))
        downsample_layers.append(d1)

        layer_count = 2
        for i in range(number_of_downsample_layers - 1):
            d = build_downsampling_block(
                downsample_layers[-1],
                filter_size=downsample_filter_window,
                stride=downsample_stride,
                tensorboard_output=tensorboard_output,
                is_training=train_flag,
                layer_number=layer_count)
            print('downsample layer: {}'.format(d.get_shape().as_list()[1:]))
            downsample_layers.append(d)
            layer_count += 1

        bn = build_downsampling_block(downsample_layers[-1],
                                      filter_size=bottleneck_filter_window,
                                      stride=bottleneck_stride,
                                      tensorboard_output=tensorboard_output,
                                      is_training=train_flag,
                                      layer_number=layer_count)
        print('bottleneck layer: {}'.format(bn.get_shape().as_list()[1:]))
        layer_count += 1

        u1 = build_upsampling_block(bn, downsample_layers[-1],
                                    depth=bn.get_shape().as_list()[-1],
                                    filter_size=upsample_filter_window,
                                    tensorboard_output=tensorboard_output,
                                    is_training=train_flag,
                                    layer_number=layer_count)
        print('upsample layer: {}'.format(u1.get_shape().as_list()[1:]))
        upsample_layers.append(u1)
        layer_count += 1

        for i in range(number_of_downsample_layers - 2, -1, -1):
            u = build_upsampling_block(upsample_layers[-1],
                                       downsample_layers[i],
                                       filter_size=upsample_filter_window,
                                       tensorboard_output=tensorboard_output,
                                       is_training=train_flag,
                                       layer_number=layer_count)
            print('upsample layer: {}'.format(u.get_shape().as_list()[1:]))
            upsample_layers.append(u)
            layer_count += 1

        target_size = int(input_size/initial_stride)
        restack = subpixel_restack(upsample_layers[-1],
                                   target_size + (upsample_filter_window - 1))
        print('restack layer: {}'.format(restack.get_shape().as_list()[1:]))

        conv = build_1d_conv_layer(restack, restack.get_shape().as_list()[-1],
                                   upsample_filter_window, initial_stride,
                                   tf.nn.elu, layer_count,
                                   padding='VALID',
                                   tensorboard_output=tensorboard_output)
        print('final conv layer: {}'.format(conv.get_shape().as_list()[1:]))

        # NOTE this effectively is a linear activation on the last conv layer
        y = subpixel_reshuffle_1D(conv,
                                  num_of_channels)
        y = tf.add(y, x, name=scope_name)
        print('output: {}'.format(y.get_shape().as_list()[1:]))

    return train_flag, x, y

# #################
# #################
