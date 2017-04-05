import numpy as np
import librosa
from inputs import get_bit_rates_and_waveforms
from inputs import get_selected_truth_ds_filename_pairs
from inputs import randomly_batch, next_batch
from inputs import read_file_pair
import tensorflow as tf
from models import single_fully_connected_model
from models import deep_residual_network


# Constants describing the training process.
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 900              # Number of epochs to train
NUM_EPOCHS_PER_DECAY = 50.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

example_number = 0
use_super_simple_model = False
waveform_reduction_factor = 1
write_tb = False
file_name_lists_dir = '/home/paperspace/Documents/EnglishSpeechUpsampler/aux'
# selected_files = ['BillGates_2010']
selected_files = ['RobertGupta_2010U']


train_truth_ds_pairs = get_selected_truth_ds_filename_pairs(
    file_name_lists_dir, selected_files, 'train')
temp_addition = get_selected_truth_ds_filename_pairs(
    file_name_lists_dir, selected_files, 'validation')
if len(temp_addition) > 0:
    train_truth_ds_pairs = train_truth_ds_pairs + temp_addition
temp_addition = get_selected_truth_ds_filename_pairs(
    file_name_lists_dir, selected_files, 'test')
if len(temp_addition) > 0:
    train_truth_ds_pairs = train_truth_ds_pairs + temp_addition

br_pairs, wf_pairs = get_bit_rates_and_waveforms(train_truth_ds_pairs[0])
true_br = br_pairs[0]
true_wf = wf_pairs[0]
waveform_max = int(true_wf.size/waveform_reduction_factor)
# true_wf = true_wf[::waveform_reduction_factor]
true_wf = true_wf[:waveform_max]
# reshape for mono waveforms
true_wf = true_wf.reshape((-1, 1))
SAMPLES_PER_EPOCH = len(train_truth_ds_pairs)
print('Samples per epoch: {}'.format(SAMPLES_PER_EPOCH))


# ################
# MODEL DEFINITION
# ################

bits_per_second = true_wf.size/10
# first_conv_depth = 128
first_conv_depth = 64
# first_conv_window = bits_per_second/3000
# first_conv_window = 30
first_conv_window = 1600
second_conv_depth = int(first_conv_depth/2)
# second_conv_depth = first_conv_depth
second_conv_window = 40
# second_conv_window = bits_per_second/4000
# thrid_conv_window = bits_per_second/6000
thrid_conv_window = 160

if use_super_simple_model:
    x, model = single_fully_connected_model(true_wf.dtype, true_wf.shape,
                                            true_wf.size, true_wf.size,
                                            write_tb)
else:
    # x, model = five_layer_conv_model(true_wf.dtype, true_wf.shape)
    train_flag, x, model = deep_residual_network(true_wf.dtype, true_wf.shape)

    # x, model = three_layer_conv_with_res_model(true_wf.dtype, true_wf.shape,
    #                                   first_conv_window, first_conv_depth,
    #                                   second_conv_window, second_conv_depth,
    #                                   thrid_conv_window,
    #                                   write_tb)
    # x, model = three_layer_conv_model(true_wf.dtype, true_wf.shape,
    #                                   first_conv_window, first_conv_depth,
    #                                   second_conv_window, second_conv_depth,
    #                                   thrid_conv_window,
    #                                   write_tb)

# placeholder for the truth label
y_true = tf.placeholder(true_wf.dtype,
                        shape=x.get_shape())

# ################
# ################


# #############
# LOSS FUNCTION
# #############

with tf.name_scope('waveform_mse'):
    waveform_mse = tf.reduce_mean(tf.square(tf.subtract(y_true, model)))
    # waveform_mse = tf.nn.l2_loss(tf.subtract(y_true, model))
    # Linf
    # waveform_mse = tf.reduce_max(tf.abs(tf.subtract(y_true, model)))
    # log-geometric mean
    # waveform_mse = tf.exp(tf.reduce_mean(tf.log1p(
    #   tf.abs(tf.subtract(y_true, model)))))
tf.summary.scalar('waveform_mse', waveform_mse)

# #############
# #############


# ####################
# OPTIMIZATION ROUTINE
# ####################

# Variables that affect learning rate.
num_batches_per_epoch = 1
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

# Decay the learning rate exponentially based on the number of steps.
global_step = tf.Variable(0, trainable=False)
with tf.name_scope('learning_rate'):
    # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                 global_step,
    #                                 decay_steps,
    #                                 LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)
    lr = tf.train.inverse_time_decay(INITIAL_LEARNING_RATE,
                                     global_step,
                                     decay_steps,
                                     LEARNING_RATE_DECAY_FACTOR,
                                     staircase=False)
tf.summary.scalar('learning_rate', lr)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    # (for batch normalization)
    with tf.name_scope('train'):
        # train_step = tf.train.RMSPropOptimizer(lr).minimize(waveform_mse,
        #                                                     global_step=global_step
        #                                                     )
        # train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(waveform_mse)
        # train_step = tf.train.AdamOptimizer(1e-4,
        #                                     epsilon=1e-01).minimize(waveform_mse)
        train_step = tf.train.AdamOptimizer(1e-4,
                                        epsilon=1e-08).minimize(waveform_mse)
        # train_step = tf.train.AdamOptimizer(lr,
        #         epsilon=1e-08).minimize(waveform_mse, global_step=global_step)
        # train_step = tf.train.AdagradOptimizer(1e-3).minimize(waveform_mse)

# ####################
# ####################


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# create session
sess = tf.Session()

# initialize tensorboard file writers
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('aux/tensorboard/overtrain',
                                     sess.graph)

# initialize the variables for the session
sess.run(tf.global_variables_initializer())


# #############
# TRAINING LOOP
# #############

example_loss = 0.0
example_loss_count = 0
for truth, example in next_batch(1, train_truth_ds_pairs):
    example_loss += np.mean((truth[0].flatten()-example[0].flatten())**2)
    example_loss_count += 1
example_loss = example_loss/float(example_loss_count)
# truth, example = read_file_pair(train_truth_ds_pairs[example_number])
# truth = truth[:waveform_max]
# example = example[:waveform_max]
# truth_batch = []
# example_batch = []
#
# for i in range(BATCH_SIZE):
#     truth_batch.append(truth)
#     example_batch.append(example)

# example_loss = np.mean((truth-example)**2)
# example_loss = 0.5*np.sum((truth-example)**2)
# example_loss = np.max(np.abs(truth-example))
# example_loss = np.exp(np.mean(np.log1p(np.abs(truth-example))))
print('loss score of example {}'.format(example_loss))
train_loss_file = open('overtrain_loss.txt', 'w')
for i in range(NUMBER_OF_EPOCHS):
    for pair in next_batch(BATCH_SIZE, train_truth_ds_pairs):
        train_step.run(feed_dict={train_flag: True,
                                  x: pair[1],
                                  y_true: pair[0]},
                       session=sess)
    batch = randomly_batch(BATCH_SIZE, train_truth_ds_pairs)
    loss_val = waveform_mse.eval(
        feed_dict={train_flag: False,
                   x: batch[1],
                   y_true: batch[0]},
        session=sess)
    print("Epoch {}, Loss {}".format((i + 1), loss_val))
    train_loss_file.write('{}\n'.format(loss_val))
    if write_tb:
        summary = sess.run([merged],
                           feed_dict={train_flag: True,
                                      x: batch[1],
                                      y_true: batch[0]})
        train_writer.add_summary(summary, i)

# for i in range(93000):
# for i in range(5000):
#     if (i + 1) % 100 == 0 or i == 0:
#         loss_val = waveform_mse.eval(
#             feed_dict={train_flag: True,
#                        x: example_batch,
#                        y_true: truth_batch},
#             session=sess)  # /float(batch_size)
#         print("Epoch {}, Loss {}".format((i + 1), loss_val))
#         train_loss_file.write('{}\n'.format(loss_val))
#     if write_tb and ((i + 1) % 500 == 0 or i == 0):
#         summary, _ = sess.run([merged, train_step],
#                               feed_dict={train_flag: True,
#                                          x: example_batch,
#                                          y_true: truth_batch})
#         train_writer.add_summary(summary, i)
#     else:
#         train_step.run(feed_dict={train_flag: True,
#                                   x: example_batch,
#                                   y_true: truth_batch},
#                        session=sess)

save_path = saver.save(sess, "aux/model_checkpoints/overtrain_final.ckpt")
print("Model checkpoints will be saved in file: {}".format(save_path))
train_loss_file.close()

truth, example = read_file_pair(train_truth_ds_pairs[example_number])
truth = truth[:waveform_max]
example = example[:waveform_max]

y_reco = model.eval(feed_dict={train_flag: True,
                               x: example.reshape(1, -1, 1)},
                    session=sess).flatten()

print('difference between truth and example (first 20 elements)')
print(truth.flatten()[:20] - example.flatten()[:20])
print('difference between truth and reconstruction (first 20 elements)')
print(truth.flatten()[:20] - y_reco[:20])

# if waveform_reduction_factor == 1:
print('writting output audio files')
librosa.output.write_wav('overtrain_true.wav',
                         y=truth.flatten(), sr=true_br)
librosa.output.write_wav('overtrain_ds.wav',
                         y=example.flatten(), sr=true_br)
librosa.output.write_wav('overtrain_reco.wav',
                         y=y_reco, sr=true_br)
# #############
# #############
