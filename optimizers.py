import tensorflow as tf


def make_variable_learning_rate(init, decay_steps,
                                decay_factor,
                                staircase=False, exp_decay=False):
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('learning_rate'):
        if exp_decay:
            lr = tf.train.exponential_decay(init, global_step,
                                            decay_steps, decay_factor,
                                            staircase=staircase)
        else:
            lr = tf.train.inverse_time_decay(init, global_step,
                                             decay_steps, decay_factor,
                                             staircase=staircase)
    tf.summary.scalar('learning_rate', lr)
    return lr, global_step


def setup_optimizer(lr, loss, opt, using_batch_norm=True,
                    opt_args={}, min_args={}):
    if using_batch_norm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing
            # the train_step (for batch normalization)
            with tf.name_scope('train'):
                train_step = opt(lr, **opt_args).minimize(loss, **min_args)
    else:
        with tf.name_scope('train'):
            train_step = opt(lr, **opt_args).minimize(loss, **min_args)
    return train_step
